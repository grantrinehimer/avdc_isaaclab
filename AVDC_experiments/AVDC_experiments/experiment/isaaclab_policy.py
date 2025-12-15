from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import sys

import cv2
import numpy as np
import torch
import random
import shutil
import math

from AVDC_experiments.flowdiffusion.flowdiffusion.inference_utils import get_video_model, pred_video

from .isaaclab_exp import utils as isaac_utils
from .myutils import (
    get_transformation_matrix,
    pred_flow_frame,
    get_transforms,
    get_transforms_from_tracks,
    sample_from_mask,
)

def move(from_xyz, to_xyz, p):
    """Computes action components that help move from 1 position to another

    Args:
        from_xyz (np.ndarray): The coordinates to move from (usually current position)
        to_xyz (np.ndarray): The coordinates to move to
        p (float): constant to scale response

    Returns:
        (np.ndarray): Response that will decrease abs(to_xyz - from_xyz)

    """
    error = to_xyz - from_xyz
    response = p * error

    if np.any(np.absolute(response) > 1.):
        warnings.warn('Constant(s) may be too high. Environments clip response to [-1, 1]')

    return response

@dataclass
class DiffusionPolicyConfig:
    """Configuration for the Isaac diffusion policy adapter."""

    camera_name: str = "overhead_camera"
    resolution: tuple[int, int] = (320, 240)
    plan_timeout: int = 15
    max_replans: int = 0
    # TODO: genuinely what should this be? it used to be 20.0
    position_gain: float = 1.0
    env_index: int = 0
    target_terms: Sequence[str] = ("Object",)
    seg_ids: Sequence[int] | None = None
    hand_body_name: str = "panda_hand"
    gripper_close_value: float = -1.0
    gripper_open_value: float = 1.0
    mode: str = "grasp"
    grasp_wait: int = 0
    sample_images: bool = False
    random_seed: int = 1
    motion_backend: str = "flow"
    locotrack_root: str | None = None
    locotrack_ckpt: str | None = None
    locotrack_model_size: str = "base"
    locotrack_query_chunk_size: int = 256
    locotrack_max_points: int = 256
    locotrack_sample_points: int = 512
    locotrack_min_points: int = 32
    diffusion_source: str = "images"
    diffusion_images_dir: str | Path | None = None


class IsaacMyPolicyCL:
    """Isaac Lab adaptation of the AVDC closed-loop policy."""

    def __init__(
        self,
        env,
        task_prompt: str,
        video_model,
        flow_model,
        config: DiffusionPolicyConfig = DiffusionPolicyConfig(),
        device: str | None = None,
        log: bool = False,
        debug: bool = False,
        save_generated: bool = False,
    ):
        self.env = env
        self.mode = config.mode
        self.task_prompt = task_prompt
        self.video_model = video_model
        self.flow_model = flow_model
        self.cfg = config
        self.motion_backend = (config.motion_backend or "flow").lower()
        if self.motion_backend not in {"flow", "locotrack"}:
            raise ValueError(f"Unsupported motion backend '{config.motion_backend}'.")
        self.camera_name = config.camera_name
        self.resolution = config.resolution
        self.plan_timeout = config.plan_timeout
        self.max_replans = config.max_replans
        self.position_gain = config.position_gain
        self.env_index = config.env_index
        self.seg_ids = config.seg_ids
        self.target_terms = config.target_terms
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log = log
        self.flow_image_path = "debug/avdc_flow_latest.png"
        self.depth_image_path = Path("debug/avdc_depth_latest.png")
        assets_root = Path(__file__).resolve().parent
        if config.diffusion_images_dir is None:
            self.diffusion_images_dir = assets_root / "diffusion_images"
        else:
            self.diffusion_images_dir = Path(config.diffusion_images_dir).expanduser()
        self.diffusion_images_dir = self.diffusion_images_dir.resolve()
        self.sample_dir = assets_root / "samples"
        self.random_seed = config.random_seed
        self.sample_images = config.sample_images
        self.diffusion_source = (config.diffusion_source or "images").lower()
        if self.diffusion_source not in {"images", "model"}:
            raise ValueError(f"Unsupported diffusion_source '{config.diffusion_source}'.")
        if self.diffusion_source == "model" and self.video_model is None:
            raise ValueError("video_model must be provided when diffusion_source='model'.")
        self.point_tracker = None
        if self.motion_backend == "flow":
            if self.flow_model is None:
                raise ValueError("flow_model must be provided when motion_backend='flow'.")
        elif self.motion_backend == "locotrack":
            self.point_tracker = self._initialize_locotrack_tracker()
        else:
            raise ValueError(f"Unknown motion backend '{self.motion_backend}'.")
        self.scene = isaac_utils.get_scene(env)
        print("env origin:", self.scene.env_origins[self.env_index])
        frame = isaac_utils.get_camera_frame(env, self.camera_name, self.env_index)
        print("camera pos_w:", frame["position"])
        self.robot = self.scene["robot"]
        if self.mode == "grasp":
            hand_ids, _ = self.robot.find_bodies(config.hand_body_name)
            if len(hand_ids) == 0:
                raise ValueError(f"Unable to resolve body named '{config.hand_body_name}'.")
            self.hand_body_id = hand_ids[0]
        
        if self.mode == "suction":
            self.ee_sensor = self.scene["ee_frame"]
            tip_ids, _ = self.ee_sensor.find_bodies("end_effector")  # matches FrameCfg.name
            if not tip_ids:
                raise RuntimeError("Unable to resolve suction_tip frame.")
            self.suction_tip_frame_id = tip_ids[0]


        self.last_pos = self._current_hand_pos()
        self.replans = self.max_replans + 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0

        self.grasp = np.zeros(3)
        self.subgoals: list[np.ndarray] = []

        self.grasped = False
        self.debug = debug
        # Suction stuff
        self.grasp_wait = config.grasp_wait
        self.grasp_count = 0
        self.save_generated=save_generated

        self._initialize_plan()

    @property
    def action_dim(self) -> int:
        return 4 if self.mode == "grasp" else 7

    def _current_hand_pos(self) -> np.ndarray:
        if self.mode == "grasp":
            pos = self.robot.data.body_pos_w[self.env_index, self.hand_body_id]
        if self.mode == "suction":
            pos = self.ee_sensor.data.target_pos_w[self.env_index, self.suction_tip_frame_id]
            print("current hand pos")
            print(pos)
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        return np.asarray(pos, dtype=np.float32)

    def _initialize_plan(self):
        grasp, transforms = self.calculate_next_plan()
        self.grasp = grasp[0]
        self.subgoals = self.calc_subgoals(self.grasp, transforms)
        self.subgoals = self._configure_mode(self.subgoals)
        self.init_grasp()

    # This is where we could configure different modes, but for now we keep it as grasp
    def _configure_mode(self, subgoals: list[np.ndarray]) -> list[np.ndarray]:
        # subgoals_np = np.array(subgoals)
        # if len(subgoals_np) > 3:
        #     max_deltaz = np.abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        # else:
        #     max_deltaz = 0.0
        # if max_deltaz > 0.1:
        #     self.mode = "grasp"
        #     return subgoals
        # self.mode = "grasp"
        # return [s - np.array([0, 0, 0.03]) for s in subgoals]
        return subgoals

    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transform in transforms:
            print("transform")
            print(transform)
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transform @ grasp_ext)[:3]
            if next_subgoal[2] < grasp[2]:
                next_subgoal[2] = grasp[2]
            subgoals.append(next_subgoal)
        print("subgoals")
        print(subgoals)
        return subgoals
        # return [np.array([0, 0, 1])]

    def _resize_rgb(self, image: np.ndarray) -> np.ndarray:
        width, height = self.resolution
        if image.shape[1] == width and image.shape[0] == height:
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        width, height = self.resolution
        if depth.shape[1] == width and depth.shape[0] == height:
            return depth
        return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

    def _output_flow_image(self, flow_images: list[np.ndarray]):
        if not flow_images:
            return
        flow_vis = flow_images[0]
        for i, flow_vis in enumerate(flow_images):
            if flow_vis is None or flow_vis.ndim != 3:
                return
            try:
                if flow_vis.shape[2] == 3:
                    img = cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR)
                else:
                    img = flow_vis
                cv2.imwrite(f"debug/avdc_flow_latest{i}.png", img)
                if self.log:
                    print(f"[Policy] optical flow image saved to {self.flow_image_path}")
            except Exception as exc:
                if self.log:
                    print(f"[Policy] failed to save optical flow image: {exc}")

    def _output_depth_image(self, depth: np.ndarray):
        if depth is None or depth.size == 0:
            return
        depth_min = np.nanmin(depth)
        depth_max = np.nanmax(depth)
        if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max - depth_min < 1e-6:
            return
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        try:
            self.depth_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.depth_image_path), depth_color)
            if self.log:
                print(f"[Policy] depth image saved to {self.depth_image_path}")
        except Exception as exc:
            if self.log:
                print(f"[Policy] failed to save depth image: {exc}")

    def _append_locotrack_root(self, root_path: str | Path):
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"LocoTrack root '{root}' does not exist.")
        if str(root) not in sys.path:
            sys.path.append(str(root))

    def _initialize_locotrack_tracker(self):
        root = self.cfg.locotrack_root
        if root:
            self._append_locotrack_root(root)
        try:
            from models.locotrack_model import load_model
        except ImportError as exc:
            raise ImportError(
                "Unable to import LocoTrack. Set 'locotrack_root' to the repository "
                "or install the package so 'models.locotrack_model' is importable."
            ) from exc
        tracker = load_model(
            ckpt_path=self.cfg.locotrack_ckpt,
            model_size=self.cfg.locotrack_model_size,
        )
        tracker = tracker.to(self.device)
        tracker.eval()
        return tracker

    def _prepare_tracker_video(self, frames: np.ndarray) -> np.ndarray:
        frames_np = np.asarray(frames)
        if frames_np.ndim != 4 or frames_np.shape[1] != 3:
            raise ValueError(
                f"Expected frames with shape (T, C, H, W), got {frames_np.shape}"
            )
        video = frames_np.transpose(0, 2, 3, 1)  # T, H, W, C
        video = video.astype(np.uint8, copy=False)
        return np.ascontiguousarray(video[np.newaxis, ...])

    def _build_tracker_queries(self, samples_2d: np.ndarray) -> torch.Tensor:
        query = np.zeros((1, len(samples_2d), 3), dtype=np.float32)
        query[0, :, 1] = samples_2d[:, 1]
        query[0, :, 2] = samples_2d[:, 0]
        return torch.from_numpy(query)

    def _track_points_with_locotrack(
        self, video: np.ndarray, query_points: torch.Tensor
    ):
        if self.point_tracker is None:
            return None, None
        resolution = (video.shape[2], video.shape[3])
        self.point_tracker.eval()
        with torch.no_grad():
            preds = self.point_tracker.inference(
                video=video,
                query_points=query_points,
                query_chunk_size=self.cfg.locotrack_query_chunk_size,
                resolution=resolution,
                query_format="tyx",
            )
        tracks = preds["tracks"].detach().cpu().numpy()
        occlusion = preds["occlusion"].detach().cpu().numpy()
        tracks = tracks[0].transpose(1, 0, 2)  # frames, points, xy
        occlusion = occlusion[0].transpose(1, 0)  # frames, points
        return tracks, occlusion

    def _tracker_motion_plan(self, images, depth, seg, cmat):
        if self.point_tracker is None:
            return None
        num_samples = max(self.cfg.locotrack_sample_points, self.cfg.locotrack_max_points)
        samples_2d = sample_from_mask(seg, num_samples).astype(np.float32)
        if samples_2d.size == 0:
            return None
        if len(samples_2d) > self.cfg.locotrack_max_points:
            samples_2d = samples_2d[: self.cfg.locotrack_max_points]
        if len(samples_2d) < self.cfg.locotrack_min_points:
            return None

        video = self._prepare_tracker_video(images)
        query_points = self._build_tracker_queries(samples_2d)

        motion_start = time.time()
        tracks, occlusion = self._track_points_with_locotrack(video, query_points)
        motion_time = time.time() - motion_start
        if tracks is None or tracks.shape[0] <= 1:
            return None

        action_start = time.time()
        grasp, transforms, _, _ = get_transforms_from_tracks(
            samples_2d,
            depth,
            cmat,
            tracks,
            occlusion=occlusion,
        )
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        action_time = time.time() - action_start
        return grasp, transform_mats, motion_time, action_time

    def _flow_motion_plan(self, images, depth, seg, cmat):
        flow_device = self.device if "cuda" in self.device else "cpu"
        start = time.time()
        _, _, flow_images, flow, _ = pred_flow_frame(self.flow_model, images, device=flow_device)
        self._output_flow_image(flow_images)
        time_motion = time.time() - start

        action_start = time.time()
        grasp, transforms, _, _ = get_transforms(seg, depth, cmat, flow)
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        time_action = time.time() - action_start
        return grasp, transform_mats, time_motion, time_action

    def _fetch_camera_observations(self):
        frame = isaac_utils.get_camera_frame(self.env, self.camera_name, self.env_index)
        image = self._resize_rgb(frame["rgb"])
        depth = self._resize_depth(frame["depth"])
        print(depth.min(), depth.max())
        cmat = isaac_utils.get_cmat(self.env, self.camera_name, self.env_index)
        seg = isaac_utils.get_seg(
            self.env,
            self.camera_name,
            resolution=self.resolution,
            seg_ids=self.seg_ids,
            target_terms=self.target_terms,
            env_index=self.env_index,
        )
        if self.debug:
            raw_seg = frame["segmentation"]
            print("seg unique IDs:", np.unique(raw_seg)[0:10])
            print("seg info keys:", frame["seg_info"])
        return image, depth, seg, cmat

    def _load_diffusion_images(self, frame_0: np.ndarray) -> np.ndarray:
        """Load cached diffusion frames, center-crop to 128x128, then pad to mirror pred_video output."""
        if not self.diffusion_images_dir.exists():
            raise FileNotFoundError(f"Diffusion image directory '{self.diffusion_images_dir}' not found.")

        image_paths = sorted(self.diffusion_images_dir.glob("*.png"))
        if len(image_paths) != 8:
            raise ValueError(
                f"Expected exactly 8 diffusion frames in '{self.diffusion_images_dir}', found {len(image_paths)}."
            )

        original_shape = frame_0.shape  # (H, W, C)
        center = (original_shape[1] // 2, original_shape[0] // 2)
        xpad = center[0] - 64
        ypad = center[1] - 64

        if xpad < 0 or ypad < 0:
            raise ValueError(
                f"Cannot pad diffusion frames to match original shape {original_shape}; resolution too small."
            )

        frames = []
        for path in image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to load diffusion frame '{path}'.")

            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Center-crop to 128x128 (no resizing / interpolation)
            h, w = img.shape[:2]
            if h < 128 or w < 128:
                raise ValueError(
                    f"Diffusion frame '{path}' is too small for a 128x128 center crop: got {w}x{h}."
                )

            top = (h - 128) // 2
            left = (w - 128) // 2
            bottom = top + 128
            right = left + 128

            img_cropped = img[top:bottom, left:right]

            # Safety check
            if img_cropped.shape[:2] != (128, 128):
                raise RuntimeError(
                    f"Center crop for '{path}' did not produce 128x128; got {img_cropped.shape[:2]}."
                )

            frames.append(img_cropped)

        # Stack into (F, H, W, C) uint8
        frames_np = np.stack(frames, axis=0).astype(np.uint8)

        # Convert to (F, C, H, W)
        frames_chw = frames_np.transpose(0, 3, 1, 2)

        # Pad back to original spatial resolution (same as pred_video)
        frames_padded = np.pad(
            frames_chw,
            pad_width=((0, 0), (0, 0), (ypad, ypad), (xpad, xpad)),
            mode="constant",
        )

        # Save debug images
        debug_dir = Path("debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i in range(frames_padded.shape[0]):
            frame = frames_padded[i].transpose(1, 2, 0)  # C, H, W -> H, W, C
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / f"diffusion_padded_{i}.png"), frame_bgr)

        return frames_padded

    def calculate_next_plan(self):
        image, depth, seg, cmat = self._fetch_camera_observations()
        self._output_depth_image(depth)

        start = time.time()
        if self.diffusion_source == "model":
            images = pred_video(self.video_model, image, self.task_prompt)
        else:
            if self.sample_images:
                print('USING SAMPLES IMAGES')
                # self._sample_and_copy_files_spaced(
                #     self.sample_dir,
                #     self.diffusion_images_dir,
                #     8,
                #     self.random_seed,
                # )
            images = self._load_diffusion_images(image)
        save_generated_path = f'AVDC_experiments/AVDC_experiments/experiment/diffusion_images/{self.task_prompt}'
        os.makedirs(save_generated_path, exist_ok=True)
        if self.save_generated:
            for img in images:
                files = os.listdir(save_generated_path)
                if len(files) == 0:
                    i = 0
                else:
                    nums = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in files])
                    i = nums[-1] + 1

                print(np.transpose(img, (1, 2, 0)).shape)
                print(f'saving to {os.path.join(save_generated_path, f"{i}.png")}')
                cv2.imwrite(
                    os.path.join(save_generated_path, f'{i}.png'),
                    np.transpose(img, (1, 2, 0)),
                )

        # images = self._load_diffusion_images(image)
        time_vid = time.time() - start
        # print(f"video model to {self.video_model.model.device}")
        # if hasattr(self.video_model, "text_encoder"):
        #     print("text encoder in use")
        #     self.video_model.text_encoder.to("cpu")

        grasp = None
        transform_mats: list[np.ndarray] | None = None
        time_motion = 0.0
        time_action = 0.0
        backend_used = self.motion_backend

        if self.motion_backend == "locotrack":
            tracker_start = time.time()
            try:
                tracker_result = self._tracker_motion_plan(images, depth, seg, cmat)
            except Exception as exc:
                tracker_result = None
                warnings.warn(f"LocoTrack planning failed, falling back to optical flow: {exc}")
            time_motion = time.time() - tracker_start
            if tracker_result is not None:
                grasp, transform_mats, time_motion, time_action = tracker_result
            else:
                backend_used = "flow"

        if transform_mats is None or grasp is None:
            backend_used = "flow"
            grasp, transform_mats, time_motion, time_action = self._flow_motion_plan(images, depth, seg, cmat)

        if self.log:
            t = max(len(transform_mats) if transform_mats else 1, 1)
            print(
                f"[Policy] plan timings (ms/frame) [{backend_used}]: "
                f"video={1000 * time_vid / t:.1f}, motion={1000 * time_motion / t:.1f}, "
                f"action={1000 * time_action / t:.1f}"
            )

        if grasp is None or transform_mats is None:
            raise RuntimeError("Motion planning failed; no transforms generated.")

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        print("grasp")
        print(grasp)
        
        # grasp = np.array([[0, 0, 0.1]])
        return grasp, transform_mats

    def init_grasp(self):
        self.grasped = False
        if self.mode == "push" and self.subgoals:
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                if norm <= 1e-6:
                    continue
                direction = (subgoal[:2] - self.grasp[:2]) / norm
                if norm > 0.1:
                    self.grasp[:2] = self.grasp[:2] - direction * 0.08
                    break

    def get_action(self, obs=None):
        pos_curr = self._current_hand_pos()
        if np.linalg.norm(pos_curr - self.last_pos) < 1e-3:
            self.replan_countdown -= 1
        self.last_pos = pos_curr.copy()
        self.time_from_last_plan += 1

        desired = self._desired_pos(pos_curr)
        delta_pos = move(pos_curr, desired, p=self.position_gain)

        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:3] = delta_pos
        action[-1] = self._grab_effort()
        return action

    def _desired_pos(self, pos_curr: np.ndarray):
        move_precision = 0.08
        grasp_precision = 0.01 if self.mode == "suction" else 0.06
        # print(self.mode)
        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            print('Would have replanned at this point')
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(self.grasp, transforms)
            self.subgoals = self._configure_mode(self.subgoals)
            if self.mode == "push":
                self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        if not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            # print("placing above object")
            return self.grasp + np.array([0.0, 0.0, 0.2])
        # drop end effector down on top of object
        if not self.grasped and abs(pos_curr[2] - self.grasp[2]) > grasp_precision:
            # print("dropping down on top of object")
            return self.grasp - np.array([0.0, 0.0, 0.01]) if self.mode == "suction" else self.grasp
        # grab object (if in grasp mode)
        if not self.grasped and abs(pos_curr[2] - self.grasp[2]) <= grasp_precision:
            print("grabbing object")
            self.grasped = True
            return self.grasp - np.array([0.0, 0.0, 0.01]) if self.mode == "suction" else self.grasp
        if self.grasped and self.grasp_count < self.grasp_wait:
            print("grasp waiting")
            self.grasp_count += 1
            return self.grasp
        # move end effector to the current subgoal
        if self.subgoals and np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            # print("moving to current subgoal")
            # print(self.subgoals[0])
            # print(np.linalg.norm(pos_curr - self.subgoals[0]))
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        if self.subgoals and len(self.subgoals) > 1:
            self.subgoals.pop(0)
            # print("moving to next subgoal")
            # print(self.subgoals[0])
            return self.subgoals[0]
        # print("executing subgoal")
        return self.subgoals[0] if self.subgoals else self.grasp

    def _grab_effort(self):
        close = self.cfg.gripper_close_value
        open_ = self.cfg.gripper_open_value
        if self.grasped or self.mode == "push":
            return close
        return open_
    
    def _sample_and_copy_files_spaced(self, src_dir, dst_dir, n, seed):
        """
        Sample n files from src_dir such that they are roughly spaced out
        across the directory (based on sorted order), then copy them to dst_dir.
        The destination directory is cleared before copying.

        Parameters:
            src_dir (str): Path to the directory to sample files from.
            dst_dir (str): Path to the directory to copy sampled files into.
            n (int): Number of files to sample.
            seed (int): Seed for random sampling.
        """
        # Ensure source directory exists
        if not os.path.isdir(src_dir):
            raise ValueError(f"Source directory does not exist: {src_dir}")

        # Create destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)

        # Clear destination directory
        for item in os.listdir(dst_dir):
            item_path = os.path.join(dst_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                shutil.rmtree(item_path)

        # Get sorted list of files (ignore subdirectories)
        files = [f for f in os.listdir(src_dir)
                if os.path.isfile(os.path.join(src_dir, f))]
        files.sort()  # important for "spaced out" sampling

        total = len(files)
        if n > total:
            raise ValueError(f"Requested {n} files, but only {total} available.")

        # Stratified sampling across the sorted list
        sampled_files = []
        chunk_size = total / n

        for i in range(n):
            start = int(math.floor(i * chunk_size))
            end = int(math.floor((i + 1) * chunk_size)) - 1
            if end < start:
                end = start
            if end >= total:
                end = total - 1

            # choose a random index from this chunk
            print(start, end)
            random.seed(seed)
            idx = random.randint(start, end)
            sampled_files.append(files[idx])

        # Copy files
        for f in sampled_files:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

        return sampled_files


def load_diffusion_video_model(ckpt_dir: str, milestone: int, flow: bool = False, timestep: int = 100):
    """Utility helper to mirror the old API when instantiating the video diffusion model."""
    return get_video_model(ckpts_dir=ckpt_dir, milestone=milestone, flow=flow, timestep=timestep)


