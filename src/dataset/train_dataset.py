import os
import os.path as osp
import random
import torch
from torch.utils.data import Dataset
from typing import List, Optional
from pathlib import Path

from src.whisper.whisper.audio import load_audio


SAMPLING_RATE = 16_000


class ProcessedDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            context_frames: int = 30,
            keypoints_fps: int = 30,
            id_list: Optional[List[int]] = None,
            *args,
            **kwargs,
        ):
        keypoints_paths = []
        audio_paths = []

        # Select id for training
        if id_list is not None:
            id_sub_dirs = [
                osp.join(root_path, id_num) for id_num in id_list]   # ["root_path/M001", ...]
        else:
            id_sub_dirs = sorted(
                [osp.join(root_path, sub_dir) for sub_dir in os.listdir(root_path)])

        # get all data path
        for id_sub_dir in id_sub_dirs:
            for pt_path in Path(id_sub_dir).rglob("*.pt"):
                # construct the corresponding audio path
                new_parts = []
                for part in pt_path.parts:
                    if part == "kp_sequences":
                        new_parts.append("audio")
                    else:
                        new_parts.append(part)

                # replace suffix to ".m4a"
                audio_path = Path(*new_parts).with_suffix(".m4a")
                if audio_path.exists():
                    keypoints_paths.append(str(pt_path))
                    audio_paths.append(str(audio_path))

        self.keypoints_paths = keypoints_paths
        self.audio_paths = audio_paths
        self.context_frames = context_frames
        self.keypoints_fps = keypoints_fps

    def __len__(self):
        return len(self.keypoints_paths)

    @torch.no_grad
    def __getitem__(self, idx):
        """Random sample a part of full sequence"""
        while True:
            idx = random.randint(0, len(self) - 1)

            audio_path = self.audio_paths[idx]
            keypoints_path = self.keypoints_paths[idx]

            keypoints = torch.load(keypoints_path, weights_only=True)
            audio = load_audio(audio_path)
            max_valid_frame = min(
                int((audio.shape[0] / SAMPLING_RATE) * self.keypoints_fps),
                keypoints.shape[0],
            )

            if max_valid_frame < self.context_frames:
                continue

            # set start frame
            start_frame = random.randint(0, max_valid_frame - self.context_frames)
            keypoints_segment = keypoints[start_frame:start_frame + self.context_frames]  # [ctx_f, 63]

            # get audio segment
            fps = self.keypoints_fps
            audio_start_index = int(start_frame / fps * SAMPLING_RATE)
            audio_segment_length = int(self.context_frames / fps * SAMPLING_RATE)

            audio_segment = audio[audio_start_index:audio_start_index + audio_segment_length]   # (16_000, ): audio signal
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)

            if audio_segment.shape[0] != audio_segment_length:
                continue

            break
        return audio_segment, keypoints_segment
