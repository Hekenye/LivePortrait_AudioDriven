from dataclasses import dataclass
from typing_extensions import Literal

from .base_config import PrintableConfig


KEYPOINTS_FPS = 30
AUDIO_FPS = 50
LIP_IDX = [6, 12, 14, 17, 19, 20]    # lip keypoints idx from LivePortrait


@dataclass(repr=False)
class SamplingConfig(PrintableConfig):
    mean: float = 0.0
    scale: float = 1.0
    min_sigma: float = 0.0
    method: Literal["euler", "dpm_solver"] = "euler"
    num_steps: int = 25
    null_condition_probability: float = 0.1


@dataclass(repr=False)
class AnimateNetConfig(PrintableConfig):
    input_dim: int = 6 * 3   # lip keypoints dim
    aud_cond_dim: int = 384      # whisper tiny feature dim
    num_heads: int = 14
    hidden_dim: int = 64 * num_heads
    depth: int = 12
    fused_depth: int = 8
    ctx_len: int = 30            # number of frames in each clip
    convert_len_multiplier: float = AUDIO_FPS / KEYPOINTS_FPS   # compute audio feature sequence len according to ctx_len
    v2: bool = True


@dataclass(repr=False)
class ModelConfig(PrintableConfig):
    animate_net_config = AnimateNetConfig()
    whisper_path: str = "./pretrained_weights/audio_processor/tiny.pt"
