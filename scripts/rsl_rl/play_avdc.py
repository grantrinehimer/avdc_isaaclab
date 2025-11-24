#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio
import numpy as np

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = (REPO_ROOT / "AVDC" / "results" / "mw").as_posix()
DEFAULT_FLOW_CKPT = (
    REPO_ROOT
    / "AVDC_experiments"
    / "AVDC_experiments"
    / "experiment"
    / "pretrained"
    / "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
).as_posix()


def _parse_args():
    parser = argparse.ArgumentParser(description="Play the AVDC diffusion policy inside Isaac Lab.")
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Custom-IK-Rel-v0", help="Name of the task.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Agent entry point (required by Hydra, not used for control).",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--camera_sensor", type=str, default="overhead_camera", help="Camera sensor name.")
    parser.add_argument("--camera_width", type=int, default=320, help="Camera width.")
    parser.add_argument("--camera_height", type=int, default=240, help="Camera height.")
    parser.add_argument("--task_prompt", type=str, default="lift", help="Text prompt for the diffusion model.")
    parser.add_argument("--plan_timeout", type=int, default=15, help="Steps before forcing a replan.")
    parser.add_argument("--max_replans", type=int, default=0, help="Maximum replans after the initial plan.")
    parser.add_argument(
        "--target_label",
        type=str,
        default="Object",
        help="Substring used to filter instance segmentation ids for the target cube.",
    )
    parser.add_argument("--num_steps", type=int, default=250, help="Maximum number of control steps to run.")
    parser.add_argument(
        "--video_ckpt_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help="Directory that stores diffusion checkpoints (expects model-<milestone>.pt).",
    )
    parser.add_argument(
        "--video_ckpt_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint file path. Overrides --video_ckpt_dir and --video_milestone.",
    )
    parser.add_argument("--video_milestone", type=int, default=24, help="Checkpoint milestone index to load.")
    parser.add_argument("--video_timestep", type=int, default=100, help="Sampling timesteps for diffusion sampling.")
    parser.add_argument("--video_flow", action="store_true", help="Load the flow-prediction diffusion model variant.")
    parser.add_argument(
        "--flow_checkpoint",
        type=str,
        default=DEFAULT_FLOW_CKPT,
        help="Path to the pretrained GMFlow checkpoint.",
    )
    parser.add_argument("--log_timings", action="store_true", help="Print per-plan timing breakdowns.")
    parser.add_argument("--save_video", action="store_true", help="Record RGB frames from the overhead camera.")
    parser.add_argument("--video_length", type=int, default=200, help="Maximum frames to save in the rollout video.")
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="Output directory for the recorded rollout video. Defaults to logs/avdc/videos.",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


def _resolve_ckpt(args_cli):
    if args_cli.video_ckpt_path:
        ckpt = Path(args_cli.video_ckpt_path).expanduser().resolve()
        stem = ckpt.stem
        if "-" in stem:
            try:
                milestone = int(stem.split("-")[-1])
            except ValueError:
                milestone = args_cli.video_milestone
        else:
            milestone = args_cli.video_milestone
        return ckpt.parent.as_posix(), milestone
    return Path(args_cli.video_ckpt_dir).expanduser().resolve().as_posix(), args_cli.video_milestone


def _ensure_video_dir(path: str | None) -> Path:
    if path is None:
        path = REPO_ROOT / "logs" / "avdc" / "videos"
    else:
        path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    args_cli, hydra_args = _parse_args()
    if args_cli.num_envs != 1:
        raise ValueError("This script currently supports num_envs=1 for closed-loop planning.")
    args_cli.enable_cameras = True
    if args_cli.save_video:
        args_cli.enable_cameras = True
    ckpt_dir, ckpt_milestone = _resolve_ckpt(args_cli)

    # store for later use
    args_cli._video_ckpt_dir = ckpt_dir
    args_cli._video_milestone = ckpt_milestone

    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab_tasks.utils.hydra import hydra_task_config

    from AVDC_experiments.experiment.isaaclab_policy import (
        DiffusionPolicyConfig,
        IsaacMyPolicyCL,
        load_diffusion_video_model,
    )
    from AVDC_experiments.experiment.myutils import get_flow_model
    from AVDC_experiments.experiment.isaaclab_exp import utils as isaac_utils

    import isaaclab_tasks  # noqa: F401
    import avdc_isaaclab.tasks  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def _run(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if hasattr(env_cfg, "camera_resolution"):
            env_cfg.camera_resolution = (args_cli.camera_width, args_cli.camera_height)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        obs, _ = env.reset()

        # video_model = load_diffusion_video_model(
        #     args_cli._video_ckpt_dir,
        #     args_cli._video_milestone,
        #     flow=args_cli.video_flow,
        #     timestep=args_cli.video_timestep,
        # )
        video_model = None
        flow_model = get_flow_model(checkpoint_path=args_cli.flow_checkpoint)

        policy_cfg = DiffusionPolicyConfig(
            camera_name=args_cli.camera_sensor,
            resolution=(args_cli.camera_width, args_cli.camera_height),
            plan_timeout=args_cli.plan_timeout,
            max_replans=0,
            seg_ids=[2],
            target_terms=(args_cli.target_label,) if args_cli.target_label else ("Object",),
        )
        policy = IsaacMyPolicyCL(
            env.unwrapped,
            args_cli.task_prompt,
            video_model,
            flow_model,
            config=policy_cfg,
            device=args_cli.device,
            log=args_cli.log_timings,
            debug=True
        )

        frames = []
        video_dir = None
        start_time = time.time()
        for step in range(1500):
            repeat = 10
            action = policy.get_action(obs)
            env_action = torch.as_tensor(action[None, :], device=env.unwrapped.device)
            for sub in range(repeat):
                obs, _, terminated, truncated, info = env.step(env_action)

            if args_cli.save_video and len(frames) < args_cli.video_length:
                frame = isaac_utils.get_camera_frame(env.unwrapped, args_cli.camera_sensor, policy.cfg.env_index)["rgb"]
                frames.append(frame.copy())

            def _any_flag(value):
                if value is None:
                    return False
                if isinstance(value, dict):
                    return any(_any_flag(v) for v in value.values())
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                return bool(np.any(np.asarray(value)))
            done = _any_flag(terminated) or _any_flag(truncated)

            if done:
                break

        elapsed = time.time() - start_time
        print(f"[INFO] Rollout finished after {step+1} steps ({elapsed:.2f}s).")

        if args_cli.save_video and frames:
            video_dir = _ensure_video_dir(args_cli.video_folder)
            video_path = video_dir / f"{Path(args_cli.task).name}-{int(time.time())}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"[INFO] Saved rollout video to {video_path}")

        env.close()

    try:
        _run()
    except Exception as e:
        print(e)
        simulation_app.close()
    finally:
        print("here")
        simulation_app.close()


if __name__ == "__main__":
    main()

