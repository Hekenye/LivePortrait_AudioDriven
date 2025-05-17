# coding: utf-8
"""
for human
"""

import tyro

from src.config.inference_with_audio_config import InferenceWithAudioConfig
from src.generator import Generator
from src.utils.misc import setup_ffmpeg


def main():
    # 1. Parse command-line arguments
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(InferenceWithAudioConfig)

    # 2. Setup FFmpeg environment
    setup_ffmpeg()

    # 3. Generate driving results
    generator = Generator(args)
    generator.inference()


if __name__ == "__main__":
    main()
