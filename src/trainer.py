import time
import os
import json
import datetime
from dataclasses import asdict

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config.train_config import TrainingArguments
from .config.model_config import KEYPOINTS_FPS
from .utils.flow_matching import FlowMatching, log_normal_sample
from .modules.animate_network import AnimateNet, filter_lip_region
from .dataset.train_dataset import ProcessedDataset
from .utils.misc import MetricStorage
from .utils.rprint import rlog as log
from .whisper.audio2feature import load_audio_model


def count_trainable_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@torch.no_grad
def get_params_norm(model: torch.nn.Module):
    params_norm_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_norm_dict[name] = param.clone().norm()
    return params_norm_dict


@torch.no_grad
def get_gradient_norm(model: torch.nn.Module):
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_dict[name] = param.grad.clone().norm()
    return grad_dict


class Trainer:
    def __init__(
            self,
            args: TrainingArguments,
    ):
        super(Trainer).__init__()
        self.args = args
        log("Loading dataset ...")

        # Init dataset
        train_dataset = ProcessedDataset(
            args.dataset_path,
            context_frames=args.model_config.animate_net_config.ctx_len,
            keypoints_fps=KEYPOINTS_FPS,
        )
        log("Dataset loaded. The dataset length is {}".format(len(train_dataset)))
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )

        # Init data processor
        self.audio_processor = load_audio_model(
            args.model_config.whisper_path,
            device=args.device,
        )

        # Init model
        statistic_path = os.path.join(args.dataset_path, "statistic.pt")
        log(f"Using keypoints' statistical information of {statistic_path}")
        model = AnimateNet(
            **asdict(args.model_config.animate_net_config),
            statistic_path=statistic_path,
        ).to(device=args.device)
        log(f"Model parameters: {count_trainable_parameters(model)} M")

        self.fm = FlowMatching(**asdict(args.sampling_config))

        # Init optimizer and lr scheduler
        optimizer, scheduler = self.configure_optimizers(model)

        # Init accelerator
        accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            project_config=accelerator_project_config
        )

        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.device = accelerator.device

        self._data_iter = iter(self.dataloader)
        self._cur_iter = 0
        self._start_iter = 0
        self._metric_storage = MetricStorage(window_size=args.log_interval)
        self._tb_writer = SummaryWriter(log_dir=os.path.join(args.work_dir, "log"))

        # save train arguments
        merged_config = {**asdict(args)}
        with open(f"{args.work_dir}/train_args.json", 'w', encoding="utf-8") as f:
            json.dump(merged_config, f, indent=4, ensure_ascii=False)

        self._last_write = {}

    def configure_optimizers(self, model: torch.nn.Module):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=[0.9, 0.95],
            fused=True,# NOTE: Try to save RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout
        )
        # linearly warmup learning rate
        linear_warmup_steps = self.args.warmup_step

        def warmup(currrent_step: int):
            return (currrent_step + 1) / (linear_warmup_steps + 1)

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)

        # setting up learning rate scheduler
        if self.args.lr_schedule == "constant":
            next_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif self.args.lr_schedule == "poly":
            total_num_iter = self.args.num_iterations
            next_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: (1 - (x / total_num_iter))**0.9)
        elif self.args.lr_schedule == "step":
            next_scheduler = MultiStepLR(optimizer, self.args.lr_schedule_steps, self.args.lr_schedule_gamma)
        else:
            raise NotImplementedError

        scheduler = SequentialLR(optimizer, [warmup_scheduler, next_scheduler], [linear_warmup_steps])
        return optimizer, scheduler

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    @property
    def ckpt_dir(self):
        return os.path.join(self.args.work_dir, "checkpoints")

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        audio, gt_kp = batch
        # select lip keypoints
        gt_kp = filter_lip_region(gt_kp)
        audio_feats = self.audio_processor.audio2feat_batch(audio, device=self.device, dtype=torch.float32)

        # classifier-free training
        bs = gt_kp.shape[0]  # (B, N, 6*3)
        samples = torch.rand(bs, device=self.device)

        # null mask is for when a video is provided but we decided to ignore it
        cfg_audio_f = audio_feats.clone()
        null_mask = (samples < self.args.sampling_config.null_condition_probability)
        if null_mask.sum() > 0:
            cfg_audio_f[null_mask] = self.model.get_empty_aud_cond(null_mask.sum())

        # sample
        x1 = self.model.normalize(gt_kp.clone())
        t = log_normal_sample(x1, m=self.args.sampling_config.mean, s=self.args.sampling_config.scale)
        x0, x1, xt = self.fm.get_x0_xt(x1, t)

        # model pred
        pred_v = self.model(xt, cfg_audio_f, t)
        mean_loss = self.fm.loss(pred_v, x0, x1).mean()
        return {
            "mse_loss": mean_loss,
        }

    def train_iter(self):
        self.model.train()
        start = time.perf_counter()
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            batch = next(self._data_iter)

        data_time = time.perf_counter() - start

        loss_dict = self.forward(batch)
        loss = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)

        params_norm_dict = get_params_norm(self.accelerator.unwrap_model(self.model))
        grad_dict = get_gradient_norm(self.accelerator.unwrap_model(self.model))

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self._log_iter_metrics(loss_dict, data_time, time.perf_counter() - start, params_norm_dict, grad_dict)

    def train(self) -> None:
        log("Start training ...")
        for self._cur_iter in range(self._start_iter + 1, self.args.max_iters + 1):
            with self.accelerator.accumulate(self.model):
                self.train_iter()

            if self._cur_iter % self.args.log_interval == 0:
                if self.accelerator.is_local_main_process:
                    self._write_console()
                    self._write_tensorboard()

            if self._cur_iter % self.args.save_interval == 0:
                self.accelerator.wait_for_everyone()
                self.save_checkpoints()

        self.accelerator.end_training()

    def _write_console(self) -> None:
        # These fields ("data_time", "iter_time", "lr", "loss") may does not
        # exist when user overwrites `Trainer.train_one_iter()`
        data_time = (self._metric_storage["data_time"].avg
                     if "data_time" in self._metric_storage else None)
        iter_time = (self._metric_storage["iter_time"].avg
                     if "iter_time" in self._metric_storage else None)
        lr = self._metric_storage["lr"].latest if "lr" in self._metric_storage else None

        if iter_time is not None:
            eta_seconds = iter_time * (self.args.max_iters - self._cur_iter)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            eta_string = None

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        loss_strings = [
            f"{key}: {his_buf.avg:.4g}" for key, his_buf in self._metric_storage.items()
            if "loss" in key and "validate" not in key
        ]

        process_string = "Iter: [{}/{}]".format(self._cur_iter,
                                                self.args.max_iters)

        space = " " * 2
        log("{process}{eta}{losses}{iter_time}{data_time}{lr}{memory}".format(
            process=process_string,
            eta=space + f"ETA: {eta_string}" if eta_string is not None else "",
            losses=space + "  ".join(loss_strings) if loss_strings else "",
            iter_time=space + f"iter_time: {iter_time:.4f}" if iter_time is not None else "",
            data_time=space + f"data_time: {data_time:.4f}" if data_time is not None else "",
            lr=space + f"lr: {lr:.5g}" if lr is not None else "",
            memory=space + f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else "",
        ))

    def _write_tensorboard(self) -> None:
        for key, (iter, value) in self._metric_storage.values_maybe_smooth.items():
            if key not in self._last_write or iter > self._last_write[key]:
                self._tb_writer.add_scalar(key, value, iter)
                self._last_write[key] = iter

    def _log_iter_metrics(
            self,
            loss_dict: dict[str, torch.Tensor],
            data_time: float,
            iter_time: float,
            params_norm_dict: dict[str, torch.Tensor]=None,
            grad_dict: dict[str, torch.Tensor]=None,
        ):
        self._metric_storage.update(self._cur_iter, lr=self.lr, smooth=False)
        self._metric_storage.update(self._cur_iter, data_time=data_time)
        self._metric_storage.update(self._cur_iter, iter_time=iter_time)
        if params_norm_dict is not None:
            for key, value in params_norm_dict.items():
                self._metric_storage.update(self._cur_iter, **{f"model_params_norm/{key}": value.detach().cpu().item()})
        if grad_dict is not None:
            for key, value in grad_dict.items():
                self._metric_storage.update(self._cur_iter, **{f"model_grad/{key}": value.detach().cpu().item()})

        loss_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        losses = sum(loss_dict.values())
        if not np.isfinite(losses):
            raise FloatingPointError(
                f"Loss became infinite or NaN at iteration={self._cur_iter + 1}! "
                f"loss_dict={loss_dict}.")

        self._metric_storage.update(self._cur_iter, **loss_dict)
        self._metric_storage.update(self._cur_iter, loss=losses)

    def save_checkpoints(self):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        log(f"Saving checkpoint to {self.ckpt_dir}")

        # Save the current state of the model, optimizer, random generators, and potentially learning rate schedulers
        save_path = os.path.join(self.ckpt_dir, f"checkpoint_{self._cur_iter}")
        self.accelerator.save_state(save_path)

        log_data = {
            "metric_storage": self._metric_storage,
        }
        file_path = os.path.join(save_path, "metric_storage.pth")
        torch.save(log_data, file_path)

    def load_checkpoints(self, path: str):
        # load state of the model, optimizer, random generators, and potentially learning rate schedulers
        self.accelerator.load_state(path)

        self._start_iter = int(path.split("_")[-1]) + 1
        self._metric_storage = torch.load(os.path.join(path, "metric_storage.pth"))["metric_storage"]
