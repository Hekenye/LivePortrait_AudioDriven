# coding: utf-8

"""
All configs for user
"""
from dataclasses import dataclass, field
from typing import Optional, Literal, List
from .base_config import PrintableConfig
from .model_config import SamplingConfig, ModelConfig


@dataclass(repr=False)  # use repr from PrintableConfig
class TrainingArguments(PrintableConfig):
    ########## training arguments ##########
    device: str = "cuda"                     # device to use
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    log_interval: int = 50                   # log period
    save_interval: int = 10000               # save period
    resume_from: Optional[str] = None        # path to the checkpoint to resume from
    auto_resume: bool = False                # whether to automatically resume from the latest checkpoint
    work_dir: str = "output"

    ########## dataset arguments ##########
    dataset_path: str = './data/processed_MEAD'  # path to the dataset root
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 8

    ########## optimizer arguments ##########
    seed: int = 0
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4         # learning rate
    max_iters: int = 30_000             # max iterations
    warmup_step: int = 1000
    lr_schedule: Literal["step", "poly", "constant"] = "step"
    lr_schedule_steps: List[int] = field(default_factory=lambda:[24_000, 27_000])
    lr_schedule_gamma: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 1.0e-6

    ########## misc arguments ##########
    sampling_config = SamplingConfig()
    model_config = ModelConfig()
