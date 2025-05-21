# coding: utf-8

"""
Wrappers for Audio driven modules
"""
import os
import os.path as osp
import torch
from einops import rearrange
from dataclasses import asdict
from safetensors.torch import load_file

from .config.model_config import LIP_IDX
from .config.inference_with_audio_config import InferenceWithAudioConfig
from .live_portrait_wrapper import LivePortraitWrapper
from .utils.cropper import Cropper
from .utils.crop import prepare_paste_back, paste_back
from .utils.helper import basename
from .utils.video import images2video, add_audio_to_video
from .utils.io import load_image_rgb, resize_to_limit
from .utils.camera import get_rotation_matrix
from .utils.flow_matching import FlowMatching
from .modules.animate_network import AnimateNet
from .utils.rprint import rlog as log
from .whisper.whisper.audio import load_audio
from .whisper.audio2feature import load_audio_model


class Generator(object):
    def __init__(
            self,
            args: InferenceWithAudioConfig,
        ):
        self.args = args
        self.device = args.device
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(
            inference_cfg=args.liveportrait_cfg,
        )
        self.cropper: Cropper = Cropper(
            crop_cfg=args.crop_cfg,
        )
        self.fm = FlowMatching(**asdict(args.sampling_config))

        # Init model
        statistic_path = args.statistic_path
        log(f"Using keypoints' statistical information of {statistic_path}")
        model = AnimateNet(
            **asdict(args.model_config.animate_net_config),
            statistic_path=statistic_path,
        ).to(device=args.device)

        # Load weight
        state_dict = load_file(args.pretrained_model_path)
        model.load_state_dict(state_dict)
        log(f'Load animate_net checkpoint from {osp.realpath(args.pretrained_model_path)} done.')
        self.model = model

        # Init data processor
        self.audio_processor = load_audio_model(
            args.model_config.whisper_path,
            device=args.device,
        )

    @torch.inference_mode()
    def inference(self):
        """Main inference pipeline"""
        source_image_path = self.args.source
        driving_audio_path = self.args.driving
        relative_motion = self.args.relative_motion
        output_dir = self.args.output_dir
        output_fps = self.args.output_fps

        log(f"Animating {os.path.relpath(source_image_path)} with audio {os.path.relpath(driving_audio_path)} ...")

        # Step 1: Process Audio
        audio_feats = self.process_audio(driving_audio_path)

        # Step 2: Crop source image
        cropped_frame, crop_info, raw_img_rgb = self.process_source_image(source_image_path)

        # Step 3: Predict keypoint sequences
        raw_pred_motions = self.predict_keypoints_motions(audio_feats)
        pred_motions = self.apply_motion_mask(raw_pred_motions, relative_motion)

        # Step 4: Generate frames using LivePortrait
        I_p_lst = self.generate_frames(cropped_frame, pred_motions)

        # Step 5: Paste back to original image size
        I_p_pstbk_lst = self.paste_back_frames(I_p_lst, crop_info, raw_img_rgb)

        # Step 6: Save result video
        output_path = self.save_video(I_p_pstbk_lst, source_image_path, driving_audio_path, output_dir, output_fps)

        return output_path

    def process_audio(self, driving_audio_path):
        """Process driving audio into features
        results:
            Audio features tensor, shape is (N_segment, seq_len, dim)
        """
        audio = torch.from_numpy(load_audio(driving_audio_path)).float().unsqueeze(0).to(self.device)
        audio_feats = self.audio_processor.audio2feat_batch(audio, device=self.device, dtype=torch.float32)

        aud_seq_len = self.model._aud_seq_len
        N_segment = audio_feats.shape[1] // aud_seq_len
        segment_start = torch.arange(0, N_segment * aud_seq_len, aud_seq_len, device=self.device)
        segment_end = segment_start + aud_seq_len

        return torch.cat([audio_feats[:, start:end] for start, end in zip(segment_start, segment_end)], dim=0)

    def process_source_image(self, src_image_path):
        """Crop and preprocess source image"""
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg
        raw_img_rgb = load_image_rgb(src_image_path)
        img_rgb = resize_to_limit(raw_img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)

        crop_info = self.cropper.crop_source_image(img_rgb, crop_cfg)
        if crop_info is None:
            raise ValueError("No face detected in the source image!")

        return crop_info['img_crop_256x256'], crop_info, raw_img_rgb

    def predict_keypoints_motions(self, audio_segment_feature):
        """Predict facial keypoints via flow matching"""
        N_segment = audio_segment_feature.shape[0]
        x0 = torch.randn(N_segment, self.model.latent_seq_len, self.model.latent_dim, device=self.device)

        preprocessed_conditions = self.model.preprocess_conditions(audio_segment_feature)
        empty_conditions = self.model.get_empty_conditions(N_segment)

        cfg_ode_wrapper = lambda t, x: self.model.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                            self.args.cfg_scale)
        x1 = self.fm.to_data(cfg_ode_wrapper, x0)
        x1 = self.model.unnormalize(x1)

        pred_motions = rearrange(x1, 'b f (n d) -> (b f) n d', b=N_segment, f=self.model.latent_seq_len, n=6, d=3)
        return pred_motions

    def apply_motion_mask(self, pred_motions, relative_motion=True):
        """
        Apply motion mask and optionally convert to relative motion.

        Args:
            pred_motions (torch.Tensor): Predicted motions of shape [T, N, D], where
                                        T = total frames, N = keypoint count (6), D = dim (3)
            relative_motion (bool): Whether to use relative motion (w.r.t. first frame)

        Returns:
            torch.Tensor: filtered motions with non-selected points zeroed out.
                        If relative_motion is True, returns delta from first frame.
        """
        # keypoints index
        if relative_motion:
            reference = pred_motions[0].unsqueeze(0)  # [1, N, D]
            pred_motions = pred_motions - reference   # delta motion related to the first frame

        # zero template
        zero_motions = torch.zeros(
            size = (pred_motions.shape[0], 21, 3),
            device = pred_motions.device,
            dtype = pred_motions.dtype,
        )
        zero_motions[:, LIP_IDX] = pred_motions
        return zero_motions

    def generate_frames(self, crooped_src_image, pred_motions):
        """Use LivePortrait to generate animated frames based on predicted keypoints"""
        I_s = self.live_portrait_wrapper.prepare_source(crooped_src_image)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)

        I_p_lst = []
        for i in range(pred_motions.shape[0]):
            delta_new = pred_motions[i].unsqueeze(0)
            scale_new = x_s_info['scale']
            t_new = x_s_info['t'].clone()
            t_new[..., 2].fill_(0)  # zero tz

            x_d_i_new = scale_new * (x_c_s @ R_s + delta_new) + t_new
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

        return I_p_lst

    def paste_back_frames(self, I_p_lst, crop_info, raw_img_rgb):
        """Paste generated frames back onto original image background"""
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        mask_ori_float = prepare_paste_back(
            inf_cfg.mask_crop,
            crop_info['M_c2o'],
            dsize=(raw_img_rgb.shape[1], raw_img_rgb.shape[0]),
        )
        return [paste_back(I_p_i, crop_info['M_c2o'], raw_img_rgb, mask_ori_float) for I_p_i in I_p_lst]

    def save_video(self, frames, src_image_path, driving_audio_path, output_dir, output_fps):
        """Save frames to video and embed audio"""
        os.makedirs(output_dir, exist_ok=True)

        base_name = f"{basename(src_image_path)}--{basename(driving_audio_path)}"
        raw_video_path = osp.join(output_dir, f"{base_name}.mp4")
        final_video_path = osp.join(output_dir, f"{base_name}_with_audio.mp4")

        # Save raw video
        images2video(frames, wfp=raw_video_path, fps=output_fps)

        # Add audio
        log(f"Adding audio from {driving_audio_path}")
        add_audio_to_video(raw_video_path, driving_audio_path, final_video_path)

        # Replace raw video with final one
        os.replace(final_video_path, raw_video_path)
        log(f"Final video saved at: {raw_video_path}")

        return raw_video_path
