#!/usr/bin/env python3

"""Collect expert camera rollouts for the lift_custom task."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
RSL_RL_DIR = SCRIPT_DIR / "rsl_rl"
try:
    from scripts.rsl_rl import cli_args  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    if str(RSL_RL_DIR) not in sys.path:
        sys.path.append(str(RSL_RL_DIR))
    import cli_args  # type: ignore  # noqa: E402


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect expert rollouts with the overhead camera.")
    parser.add_argument(
        "--task", type=str, default="Isaac-Lift-Cube-Franka-Custom-IK-Rel-v0", help="Gym task name."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="RL agent configuration entry point registered with the task.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
    parser.add_argument("--num_rollouts", type=int, default=1, help="How many episodes to record.")
    parser.add_argument("--max_steps", type=int, default=250, help="Maximum steps per rollout.")
    parser.add_argument("--dataset_root", type=str, default="AVDC/datasets/isaaclab", help="Base dataset directory.")
    parser.add_argument("--dataset_task", type=str, default="lift_custom", help="Task folder inside metaworld_dataset.")
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="overhead",
        help="Variant folder inside the task directory (e.g., camera viewpoint).",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Episode index to start numbering from.")
    parser.add_argument("--frame_digits", type=int, default=2, help="Zero-padding for frame filenames.")
    parser.add_argument("--camera_sensor", type=str, default="overhead_camera", help="Camera sensor name in the scene.")
    parser.add_argument("--camera_env_index", type=int, default=0, help="Environment index to read frames from.")
    parser.add_argument("--camera_width", type=int, default=320, help="Camera image width override.")
    parser.add_argument("--camera_height", type=int, default=240, help="Camera image height override.")
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=False,
        help="If set, dump a metadata.json file alongside each action.pkl.",
    )
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        default=False,
        help="Download and use the published RSL-RL checkpoint (matches play.py behavior).",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def capture_frame(sensor, env_index: int) -> np.ndarray:
    """Fetch an RGB frame (uint8, H x W x 3) for a specific environment index."""
    data = sensor.data
    if "rgb" not in data.output:
        raise RuntimeError("Camera sensor is not publishing RGB data. Enable rgb in CameraCfg.data_types.")
    rgb_tensor = data.output["rgb"]
    frame = rgb_tensor[env_index].detach().cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def ensure_sequence_dir(base_dir: Path, start_index: int) -> tuple[Path, int]:
    """Create a new sequence directory, ensuring we don't overwrite existing rollouts."""
    idx = start_index
    while True:
        seq_dir = base_dir / f"{idx:03d}"
        if not seq_dir.exists():
            seq_dir.mkdir(parents=True, exist_ok=False)
            return seq_dir, idx + 1
        idx += 1


def save_action_file(path: Path, actions: list[np.ndarray], success: bool):
    payload: dict[str, Any] = {
        "actions": np.stack(actions, axis=0) if actions else np.empty((0,), dtype=np.float32),
        "num_steps": len(actions),
        "success": success,
    }
    with path.open("wb") as fp:
        pickle.dump(payload, fp)


def save_metadata_file(path: Path, info: dict[str, Any]):
    import json

    with path.open("w", encoding="utf-8") as fp:
        json.dump(info, fp, indent=2)


def main():
    parser = make_parser()
    args_cli, hydra_args = parser.parse_known_args()

    if args_cli.num_envs != 1:
        raise ValueError("This data collector currently supports num_envs=1 for deterministic rollouts.")

    # Always render RTX sensors for RGB capture
    args_cli.enable_cameras = True

    # Pass Hydra the remaining args
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Lazy imports after the simulator starts
    import gymnasium as gym
    import torch
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_tasks  # noqa: F401
    import avdc_isaaclab.tasks  # noqa: F401

    dataset_root = Path(args_cli.dataset_root).expanduser().resolve()
    dataset_base = dataset_root / args_cli.dataset_task / args_cli.dataset_variant
    dataset_base.mkdir(parents=True, exist_ok=True)

    @hydra_task_config(args_cli.task, args_cli.agent)
    def _run(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlBaseRunnerCfg,
    ):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if hasattr(env_cfg, "camera_resolution"):
            env_cfg.camera_resolution = (args_cli.camera_width, args_cli.camera_height)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Locate checkpoint
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        published_task_name = train_task_name.replace("-Custom", "")
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()

        if args_cli.use_pretrained_checkpoint:
            checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", published_task_name)
            if not checkpoint_path:
                raise RuntimeError(
                    f"No published checkpoint found for task '{published_task_name}'. "
                    "Disable --use_pretrained_checkpoint or provide --checkpoint."
                )
        elif args_cli.checkpoint is not None:
            checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        else:
            checkpoint_path = get_checkpoint_path(str(log_root_path), agent_cfg.load_run, agent_cfg.load_checkpoint)
            if not checkpoint_path:
                checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", published_task_name)
                if not checkpoint_path:
                    raise RuntimeError(
                        "Unable to locate a trained checkpoint. "
                        "Train a policy first or rerun with --use_pretrained_checkpoint/--checkpoint."
                    )

        env_cfg.log_dir = os.path.dirname(checkpoint_path)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=vec_env.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        camera_sensor = getattr(vec_env.unwrapped.scene, "sensors", {}).get(args_cli.camera_sensor)
        if camera_sensor is None:
            raise RuntimeError(
                f"Camera sensor '{args_cli.camera_sensor}' not found. "
                "Verify that LiftEnvCfg.scene.overhead_camera is enabled."
            )

        seq_index = args_cli.start_index
        obs = vec_env.get_observations()

        for rollout_idx in range(args_cli.num_rollouts):
            seq_dir, seq_index = ensure_sequence_dir(dataset_base, seq_index)
            frame_paths: list[Path] = []
            actions_buffer: list[np.ndarray] = []

            # initial frame before taking any action
            frame = capture_frame(camera_sensor, args_cli.camera_env_index)
            frame_path = seq_dir / f"{0:0{args_cli.frame_digits}d}.png"
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)

            success = False
            for step in range(1, args_cli.max_steps + 1):
                with torch.inference_mode():
                    actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                policy_nn.reset(dones)
                actions_buffer.append(actions[0].detach().cpu().numpy())
                if not dones[0].item():
                    frame = capture_frame(camera_sensor, args_cli.camera_env_index)
                    frame_path = seq_dir / f"{step:0{args_cli.frame_digits}d}.png"
                    Image.fromarray(frame).save(frame_path)
                    frame_paths.append(frame_path)
                else:
                    success = True
                    break

            save_action_file(seq_dir / "action.pkl", actions_buffer, success)
            if args_cli.metadata:
                save_metadata_file(
                    seq_dir / "metadata.json",
                    {
                        "task": args_cli.task,
                        "camera_sensor": args_cli.camera_sensor,
                        "camera_resolution": [args_cli.camera_width, args_cli.camera_height],
                        "checkpoint": str(checkpoint_path),
                        "num_frames": len(frame_paths),
                        "num_actions": len(actions_buffer),
                        "success": success,
                    },
                )

            print(
                f"[Rollout {rollout_idx+1}/{args_cli.num_rollouts}] "
                f"Saved {len(frame_paths)} frames to {seq_dir} (success={success})."
            )

        vec_env.close()

    try:
        _run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
