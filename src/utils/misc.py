import os
import random
import subprocess
import torch
import numpy as np
from typing import Optional
from collections import deque
from typing import Optional

from .rprint import rlog as log


def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): If None or negative, use a generated seed.
        deterministic (bool): If True, set the deterministic option for CUDNN backend.
    """
    if seed is None or seed < 0:
        new_seed = np.random.randint(2**32)
        log(f"Got invalid seed: {seed}, will use the randomly generated seed: {new_seed}")
        seed = new_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        log("The CUDNN is set to deterministic. This will increase reproducibility, "
                    "but may slow down your training considerably.")


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def setup_ffmpeg() -> None:
    """Add local ffmpeg binary directory to PATH if it exists."""
    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if os.path.exists(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe)" \
            " before running this script. https://ffmpeg.org/download.html"
        )


class HistoryBuffer:
    """
    This class tracks a series data and provides statistics about it.
    """
    def __init__(self, window_size: int=50):
        self._history = deque(maxlen=window_size)
        self._count: int = 0
        self._sum: float = 0.

    def update(self, data: float):
        self._history.append(data)
        self._count += 1
        self._sum += data

    @property
    def avg(self):
        return np.mean(self._history)

    @property
    def latest(self):
        return self._history[-1]


class MetricStorage(dict):
    """The class stores the values of multiple metrics (some of them may be noisy, e.g., loss,
    batch time) in training process, and provides access to the smoothed values for better logging.

    The class is designed for automatic tensorboard logging. User should specify the ``smooth``
    when calling :meth:`update`, so that we can determine which metrics should be
    smoothed when performing tensorboard logging.

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2)
        >>> metric_storage.update(iter=0, lr=0.01, smooth=False)
        >>> metric_storage.update(iter=1, loss=0.1)
        >>> metric_storage.update(iter=1, lr=0.001, smooth=False)
        >>> # loss will be smoothed, but lr will not
        >>> metric_storage.values_maybe_smooth
        {"loss": (1, 0.15), "lr": (1, 0.001)}
        >>> # like dict, can be indexed by string
        >>> metric_storage["loss"].avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._history: dict[str, HistoryBuffer] = self
        self._smooth: dict[str, bool] = {}
        self._latest_iter: dict[str, int] = {}

    def update(self, iter: Optional[int] = None, smooth: bool = True, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration.

        Args:
            iter (int): The iteration in which these values are produced.
                If None, use the built-in counter starting from 0.
            smooth (bool): If True, return the smoothed values of these metrics when
                calling :meth:`values_maybe_smooth`. Otherwise, return the latest values.
                The same metric must have the same ``smooth`` in different calls to :meth:`update`.
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                assert self._smooth[key] == smooth
            else:
                self._smooth[key] = smooth
                self._history[key] = HistoryBuffer(window_size=self._window_size)
                self._latest_iter[key] = -1
            if iter is not None:
                assert iter > self._latest_iter[key]
                self._latest_iter[key] = iter
            else:
                self._latest_iter[key] += 1
            self._history[key].update(value)

    @property
    def values_maybe_smooth(self):
        """Return the smoothed values or the latest values of multiple metrics.
        The specific behavior depends on the ``smooth`` when updating metrics.

        Returns:
            dict[str -> (int, float)]:
                Mapping from metric name to its (the latest iteration, the avg / the latest value)
                pair.
        """
        return {
            key: (self._latest_iter[key], his_buf.avg if self._smooth[key] else his_buf.latest)
            for key, his_buf in self._history.items()
        }
