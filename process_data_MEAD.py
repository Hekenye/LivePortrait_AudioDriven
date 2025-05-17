import argparse
import os
import os.path as osp
import yaml
import shutil
import torch
from typing import List, Optional
from torchvision import transforms
from decord import VideoReader
from tqdm import tqdm

from src.utils.rprint import rlog as log
from src.modules.motion_extractor import MotionExtractor
from src.config.crop_config import CropConfig
from src.utils.cropper import Cropper


EMOTION_TYPES = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_LEVELS = [1]
VIDEO_ANGLES = ["front"]


class Processor:
    def __init__(
        self,
        root_path: str,
        liveportrait_modules_config: str,
        motion_extractor_path: str,
        image_size: int = 256,
        device: str = "cuda",
        *,
        id_list: Optional[List[str]] = None,
        emotion_list: Optional[List[str]] = None,
        emotion_level_list: Optional[List[int]] = None,
        video_angle_list: Optional[List[str]] = None,
    ):
        if id_list is not None:
            subject_paths = [osp.join(root_path, sid) for sid in id_list]
        else:
            subject_paths = sorted([osp.join(root_path, d) for d in os.listdir(root_path)])

        emotion_list = emotion_list or EMOTION_TYPES
        emotion_level_list = [f"level_{i}" for i in (emotion_level_list or EMOTION_LEVELS)]
        video_angle_list = video_angle_list or VIDEO_ANGLES

        log(f"Dataset Subjects: {[osp.basename(p) for p in subject_paths]}")
        log(f"Emotion Types: {emotion_list}")
        log(f"Emotion Levels: {emotion_level_list}")
        log(f"Video Angles: {video_angle_list}")

        # collect all video and audio paths
        video_paths, audio_paths = [], []
        total_tasks = len(subject_paths) * len(video_angle_list) * len(emotion_list) * len(emotion_level_list)
        progress_bar = tqdm(range(total_tasks), desc="Loading MEAD dataset")

        for subject_dir in subject_paths:
            for angle in video_angle_list:
                for emotion in emotion_list:
                    for level in emotion_level_list:
                        emotion_dir = f"{emotion}/{level}"
                        video_dir = osp.join(subject_dir, "video", angle, emotion_dir)
                        audio_dir = osp.join(subject_dir, "audio", emotion_dir)

                        if not osp.exists(video_dir):
                            progress_bar.update(1)
                            continue

                        for video_name in sorted(os.listdir(video_dir)):
                            if not video_name.endswith(".mp4"):
                                continue
                            audio_name = video_name.replace("mp4", "m4a")
                            video_path = osp.join(video_dir, video_name)
                            audio_path = osp.join(audio_dir, audio_name)

                            if osp.exists(audio_path):
                                video_paths.append(video_path)
                                audio_paths.append(audio_path)

                        progress_bar.update(1)
        progress_bar.close()

        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.root_path = root_path

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size))
        ])

        # init motion extractor
        with open(liveportrait_modules_config, 'r') as f:
            model_cfg = yaml.safe_load(f)['model_params']['motion_extractor_params']
        self.motion_extractor = MotionExtractor(**model_cfg).to(device)
        self.motion_extractor.load_state_dict(
            torch.load(motion_extractor_path, map_location='cpu', weights_only=True)
        )
        self.device = device

        # init cropper
        self.cropper = Cropper(crop_cfg=CropConfig(dsize=image_size))

    def process_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Extract motion keypoints from single video"""
        vr = VideoReader(video_path)
        frames = []

        # read frame and crop face region
        for frame_idx in range(len(vr)):
            frame = vr[frame_idx].asnumpy()
            crop_result = self.cropper.crop_source_image(frame, self.cropper.crop_cfg)
            if crop_result is None:
                log(f"No face detected in a frame from {video_path}. Skipping.")
                return None
            cropped_frame = crop_result['img_crop_256x256']
            tensor_frame = self.image_transform(cropped_frame)
            frames.append(tensor_frame)

        if not frames:
            log(f"No valid frames with face detected in video: {video_path}")
            return None

        # extract keypoints
        frames_tensor = torch.stack(frames, dim=0).to(self.device)
        batch_size = 160
        keypoints_deltas = []

        for i in range(0, len(frames_tensor), batch_size):
            batch = frames_tensor[i:i + batch_size]
            output = self.motion_extractor(batch)
            keypoints_deltas.append(output["exp"])

        return torch.cat(keypoints_deltas, dim=0)


    @torch.inference_mode()
    def run(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        all_keypoints_deltas = []

        for idx in tqdm(range(len(self.video_paths)), desc="Processing videos"):
            try:
                video_path = self.video_paths[idx]
                keypoints_deltas = self.process_video(video_path)
                if keypoints_deltas is None:
                    continue

                all_keypoints_deltas.append(keypoints_deltas)

                # output path
                parts = video_path.split("/")
                filename = parts[-1].split(".")[0]
                subject_id = parts[-6]
                emotion_type = parts[-3]
                emotion_level = parts[-2]
                out_subdir = osp.join(output_dir, subject_id, emotion_type, emotion_level, "kp_sequences")
                os.makedirs(out_subdir, exist_ok=True)
                out_path = osp.join(out_subdir, f"{filename}.pt")

                torch.save(keypoints_deltas.cpu(), out_path)

                # copy audio file
                audio_path = self.audio_paths[idx]
                audio_name = osp.basename(audio_path)
                audio_out_dir = out_subdir.replace("kp_sequences", "audio")
                os.makedirs(audio_out_dir, exist_ok=True)
                shutil.copy2(audio_path, osp.join(audio_out_dir, audio_name))

            except Exception as e:
                log(f"Error processing {video_path}: {e}")
                continue

        # statistic mean / std
        if all_keypoints_deltas:
            all_kps_tensor = torch.cat(all_keypoints_deltas, dim=0)
            stat = {"mean": torch.mean(all_kps_tensor, dim=0), "std": torch.std(all_kps_tensor, dim=0)}
            torch.save(stat, osp.join(output_dir, "statistic.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The root path of your MEAD dataset.")
    parser.add_argument("--output_path", type=str, default="./data/processed_MEAD", help="The path of save your process results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    root_path = args.dataset_path
    output_path = args.output_path
    liveportrait_modules_config = "./src/config/models.yaml"
    motion_extractor_path: str = './pretrained_weights/liveportrait/base_models/motion_extractor.pth'  # path to checkpoint of motion extractor
    image_size: int = 256

    id_list = ["M003"]
    emotion_list = ["angry"]
    emotion_level_list = [1]
    video_angle_list = ["front"]

    runner = Processor(
        root_path,
        liveportrait_modules_config,
        motion_extractor_path,
        image_size,
        id_list=id_list,
        emotion_list=emotion_list,
        emotion_level_list=emotion_level_list,
        video_angle_list=video_angle_list,
    )
    runner.run(output_dir=output_path)
