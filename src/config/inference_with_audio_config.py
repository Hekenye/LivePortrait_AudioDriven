# coding: utf-8

"""
All configs for user
"""
from dataclasses import dataclass
import tyro
from typing_extensions import Annotated, Literal

from .base_config import PrintableConfig
from .model_config import ModelConfig, SamplingConfig
from .inference_config import InferenceConfig
from .crop_config import CropConfig


@dataclass(repr=False)  # use repr from PrintableConfig
class InferenceWithAudioConfig(PrintableConfig):
    ########## inference arguments ##########
    source: Annotated[str, tyro.conf.arg(aliases=["-s"])]  # path to the source portrait(human)
    driving:  Annotated[str, tyro.conf.arg(aliases=["-d"])]  # path to driving audio
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'generated/'  # directory to save output video
    pretrained_model_path: str = None        # path to the pretrained checkpoint
    device: str = "cuda"                     # device to use
    seed: int = 0
    cfg_scale: float = 4.5
    output_fps: int = 30
    relative_motion: bool = True

    ########## misc arguments ##########
    sampling_config = SamplingConfig()
    model_config = ModelConfig()
    liveportrait_cfg = InferenceConfig()
    crop_cfg = CropConfig()
    statistic_path: str = None
