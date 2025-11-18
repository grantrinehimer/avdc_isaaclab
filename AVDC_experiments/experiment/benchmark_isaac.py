"""Scaffold for running MyPolicy_CL inside the IsaacLab cs6758_custom task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from env_interfaces.isaaclab import build_env_adapter
from flowdiffusion.inference_utils import get_video_model
from mypolicy import MyPolicy_CL
from myutils import get_flow_model


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MyPolicy_CL inside IsaacLab.")
    parser.add_argument(
        "--task",
        type=str,
        default="avdc_isaaclab::cs6758_custom/FrankaBasketCubeEnv",
        help="Gym registration string for the Isaac environment.",
    )
    parser.add_argument(
        "--policy_task",
        type=str,
        default="door-open-v2-goal-observable",
        help="Key used in name2maskid.json for segmentation lookup.",
    )
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=24)
    parser.add_argument("--camera", type=str, default="corner")
    parser.add_argument("--resolution", type=int, nargs=2, default=(320, 240))
    parser.add_argument("--n_exps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--disable_fabric", action="store_true")
    parser.add_argument("--result_root", type=str, default="../results/isaac")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _collect_episode(adapter, policy, camera_name, resolution, max_steps=1024, seed: int | None = None):
    """Mimic collect_video but route all perception through the EnvAdapter."""

    images: List[np.ndarray] = []
    depths: List[np.ndarray] = []

    reset_out = adapter.reset(seed=seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    reward_total = 0.0
    done = False
    steps = 0

    try:
        rgb, depth = adapter.fetch_rgbd(camera_name, resolution)
        images.append(rgb)
        depths.append(depth)
    except NotImplementedError as exc:
        raise RuntimeError(
            "IsaacLabEnvAdapter.fetch_rgbd is still unimplemented. "
            "See cs6758_custom/TODO.md for the remaining wiring steps."
        ) from exc

    while not done and steps < max_steps:
        action = policy.get_action(obs)
        step_out = adapter.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        reward_total += reward
        steps += 1

        # TODO: once sensors are available, capture RGB-D after each step for logging.
        try:
            rgb, depth = adapter.fetch_rgbd(camera_name, resolution)
            images.append(rgb)
            depths.append(depth)
        except NotImplementedError:
            pass

    return images, depths, reward_total


def main():
    args = _parse_args()
    os.makedirs(args.result_root, exist_ok=True)

    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()

    camera_setup = {args.camera: tuple(args.resolution)}
    adapter = build_env_adapter(
        task=args.task,
        camera_setup=camera_setup,
        device=args.device,
        num_envs=args.num_envs,
        disable_fabric=args.disable_fabric,
        app_launcher_kwargs={"headless": args.headless},
    )

    env = adapter.raw_env
    policy = MyPolicy_CL(
        env,
        args.policy_task,
        args.camera,
        video_model,
        flow_model,
        resolution=tuple(args.resolution),
        max_replans=5,
        env_adapter=adapter,
    )

    rewards = []
    for seed in tqdm(range(args.n_exps)):
        images, depths, reward = _collect_episode(
            adapter,
            policy,
            args.camera,
            tuple(args.resolution),
            seed=seed,
        )
        rewards.append(reward)

        with open(Path(args.result_root) / f"episode_{seed}.json", "w") as f:
            json.dump({"reward": reward, "num_frames": len(images)}, f, indent=2)

    print("Mean reward:", np.mean(rewards))
    adapter.close()


if __name__ == "__main__":
    main()

