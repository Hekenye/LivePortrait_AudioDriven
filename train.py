# -*- coding: utf-8 -*-
import os
from typing import Optional

import tyro

from src.config.train_config import TrainingArguments
from src.trainer import Trainer
from src.utils.rprint import rlog as log
from src.utils.misc import set_random_seed, setup_ffmpeg


def auto_resume_from_latest_checkpoint(work_dir: str) -> Optional[str]:
    """Check for latest checkpoint in work_dir/checkpoints and return its path if exists."""
    ckpt_dir = os.path.join(work_dir, "checkpoints")

    if not os.path.exists(ckpt_dir):
        log(f"Auto-resume enabled but checkpoint directory '{ckpt_dir}' does not exist.")
        return None

    ckpt_list = sorted(os.listdir(ckpt_dir))
    if not ckpt_list:
        log(f"Auto-resume enabled but checkpoint directory '{ckpt_dir}' is empty.")
        return None

    latest_ckpt = os.path.join(ckpt_dir, ckpt_list[-1])
    log(f"Found latest checkpoint: {latest_ckpt}")
    return latest_ckpt


def main() -> None:
    # 1. Parse command-line arguments
    tyro.extras.set_accent_color("bright_cyan")
    args: TrainingArguments = tyro.cli(TrainingArguments)

    # 2. Setup FFmpeg environment
    setup_ffmpeg()

    # 3. Set random seed and initialize trainer
    set_random_seed(args.seed)
    log(args)
    trainer = Trainer(args)

    # 4. Determine resume path
    resume_path: Optional[str] = args.resume_from
    if args.auto_resume:  # auto_resume has a higher priority than resume_from
        resume_path = auto_resume_from_latest_checkpoint(args.work_dir)

    # 5. Load checkpoint if specified
    if resume_path:
        log(f"Loading checkpoint from: {resume_path}")
        trainer.load_checkpoints(resume_path)

    # 6. Start training
    trainer.train()


if __name__ == "__main__":
    main()
