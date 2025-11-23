from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch


from AVDC_experiments.flowdiffusion.flowdiffusion.inference_utils import get_video_model, pred_video

from .isaaclab_exp import utils as isaac_utils
from .myutils import get_transformation_matrix, pred_flow_frame, get_transforms

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
    ):
        self.env = env
        self.task_prompt = task_prompt
        self.video_model = video_model
        self.flow_model = flow_model
        self.cfg = config
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
        self.flow_image_path = "/tmp/avdc_flow_latest.png"
        self.diffusion_images_dir = Path(__file__).resolve().parent / "diffusion_images"

        self.scene = isaac_utils.get_scene(env)
        self.robot = self.scene["robot"]
        hand_ids, _ = self.robot.find_bodies(config.hand_body_name)
        if len(hand_ids) == 0:
            raise ValueError(f"Unable to resolve body named '{config.hand_body_name}'.")
        self.hand_body_id = hand_ids[0]

        self.last_pos = self._current_hand_pos()
        self.replans = self.max_replans + 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0

        self.grasp = np.zeros(3)
        self.subgoals: list[np.ndarray] = []
        self.mode = "grasp"
        self.grasped = False

        self._initialize_plan()

    @property
    def action_dim(self) -> int:
        return 7  # 6D pose command + binary gripper

    def _current_hand_pos(self) -> np.ndarray:
        pos = self.robot.data.body_pos_w[self.env_index, self.hand_body_id]
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        return np.asarray(pos, dtype=np.float32)

    def _initialize_plan(self):
        grasp, transforms = self.calculate_next_plan()
        self.grasp = grasp[0]
        self.subgoals = self.calc_subgoals(self.grasp, transforms)
        self.subgoals = self._configure_mode(self.subgoals)
        self.init_grasp()

    def _configure_mode(self, subgoals: list[np.ndarray]) -> list[np.ndarray]:
        subgoals_np = np.array(subgoals)
        if len(subgoals_np) > 3:
            max_deltaz = np.abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        else:
            max_deltaz = 0.0
        if max_deltaz > 0.1:
            self.mode = "grasp"
            return subgoals
        self.mode = "push"
        return [s - np.array([0, 0, 0.03]) for s in subgoals]

    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transform in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transform @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

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
        if flow_vis is None or flow_vis.ndim != 3:
            return
        try:
            if flow_vis.shape[2] == 3:
                img = cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR)
            else:
                img = flow_vis
            cv2.imwrite(self.flow_image_path, img)
            if self.log:
                print(f"[Policy] optical flow image saved to {self.flow_image_path}")
        except Exception as exc:
            if self.log:
                print(f"[Policy] failed to save optical flow image: {exc}")

    def _fetch_camera_observations(self):
        frame = isaac_utils.get_camera_frame(self.env, self.camera_name, self.env_index)
        image = self._resize_rgb(frame["rgb"])
        depth = self._resize_depth(frame["depth"])
        cmat = isaac_utils.get_cmat(self.env, self.camera_name, self.env_index)
        seg = isaac_utils.get_seg(
            self.env,
            self.camera_name,
            resolution=self.resolution,
            seg_ids=self.seg_ids,
            target_terms=self.target_terms,
            env_index=self.env_index,
        )
        return image, depth, seg, cmat

    def _load_diffusion_images(self, frame_0: np.ndarray) -> np.ndarray:
        """Load cached diffusion frames and pad them to mirror pred_video output."""
        if not self.diffusion_images_dir.exists():
            raise FileNotFoundError(f"Diffusion image directory '{self.diffusion_images_dir}' not found.")

        image_paths = sorted(self.diffusion_images_dir.glob("*.png"))
        if len(image_paths) != 8:
            raise ValueError(
                f"Expected exactly 8 diffusion frames in '{self.diffusion_images_dir}', found {len(image_paths)}."
            )

        frames = []
        for path in image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to load diffusion frame '{path}'.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (128, 128):
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            frames.append(img)

        frames_np = np.stack(frames, axis=0).astype(np.uint8)

        original_shape = frame_0.shape
        center = (original_shape[1] // 2, original_shape[0] // 2)
        xpad = center[0] - 64
        ypad = center[1] - 64
        if xpad < 0 or ypad < 0:
            raise ValueError(
                f"Cannot pad diffusion frames to match original shape {original_shape}; resolution too small."
            )

        frames_chw = frames_np.transpose(0, 3, 1, 2)
        frames_padded = np.pad(frames_chw, ((0, 0), (0, 0), (ypad, ypad), (xpad, xpad)), mode="constant")
        return frames_padded

    def calculate_next_plan(self):
        image, depth, seg, cmat = self._fetch_camera_observations()

        start = time.time()
        # images = pred_video(self.video_model, image, self.task_prompt)
        images = self._load_diffusion_images(image)
        print(images.shape)
        time_vid = time.time() - start
        # print(f"video model to {self.video_model.model.device}")
        # if hasattr(self.video_model, "text_encoder"):
        #     print("text encoder in use")
        #     self.video_model.text_encoder.to("cpu")

        start = time.time()
        _, _, flow_images, flow, _ = pred_flow_frame(self.flow_model, images, device="cuda:0")
        self._output_flow_image(flow_images)
        time_flow = time.time() - start

        start = time.time()
        grasp, transforms, _, _ = get_transforms(seg, depth, cmat, flow)
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        time_action = time.time() - start

        if self.log:
            t = max(len(transform_mats), 1)
            print(
                f"[Policy] plan timings (ms/frame): "
                f"video={1000 * time_vid / t:.1f}, flow={1000 * time_flow / t:.1f}, action={1000 * time_action / t:.1f}"
            )

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
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
        delta_pos = np.clip(delta_pos, -1.0, 1.0)

        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:3] = delta_pos
        action[6] = self._grab_effort()
        return action

    def _desired_pos(self, pos_curr: np.ndarray):
        move_precision = 0.12 if self.mode == "push" else 0.04

        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(self.grasp, transforms)
            self.subgoals = self._configure_mode(self.subgoals)
            if self.mode == "push":
                self.init_grasp()
            return self.subgoals[0]
        if not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0.0, 0.0, 0.2])
        if not self.grasped and abs(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        if not self.grasped and abs(pos_curr[2] - self.grasp[2]) <= 0.04:
            print("grasped")
            self.grasped = True
            return self.grasp
        if self.subgoals and np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        if self.subgoals and len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        return self.subgoals[0] if self.subgoals else self.grasp

    def _grab_effort(self):
        close = self.cfg.gripper_close_value
        open_ = self.cfg.gripper_open_value
        if self.grasped or self.mode == "push":
            return close
        return open_


def load_diffusion_video_model(ckpt_dir: str, milestone: int, flow: bool = False, timestep: int = 100):
    """Utility helper to mirror the old API when instantiating the video diffusion model."""
    return get_video_model(ckpts_dir=ckpt_dir, milestone=milestone, flow=flow, timestep=timestep)

