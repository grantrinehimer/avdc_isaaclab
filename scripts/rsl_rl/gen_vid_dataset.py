# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import shutil
import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate an AVDC-format video dataset for training the diffusion model.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--n_rollouts", type=int, default=8, help="Minimum number of rollouts to collect for each command (target pose).")
parser.add_argument("--task_shorthand", type=str, default="Lift-Cube-Randomized", help="Shorthand task name. Use either Lift-Cube-Randomized or Lift-Cube-Deterministc")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
args_cli.enable_cameras = True

# Task lookup based on shorthand. These are the "with camera" tasks from the registry.
if args_cli.task_shorthand == "Lift-Cube-Randomized":
    args_cli.task = "Isaac-Lift-Cube-Franka-Dataset-Randomized-v0"
elif args_cli.task_shorthand == "Lift-Cube-Deterministic":
    args_cli.task = "Isaac-Lift-Cube-Franka-Dataset-Deterministic-v0"
else:
    raise ValueError("Please enter a valid task shorthand.")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from isaaclab.sensors import Camera, CameraCfg

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import avdc_isaaclab.tasks  # noqa: F401
from AVDC_experiments.experiment.isaaclab_exp import utils as isaac_utils



# Dataset writer to format video recordings like an AVDC training dataset
class dataset_writer:
    def __init__(self, root, num_envs, n_rollouts, n_commands, command_ranges):
        self.root = root    # dataset root e.g., ./AVDC/datasets/isaaclab/LiftCubeRandomized
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.mkdir(root)
        self.num_envs = num_envs # number of parallel IsaacLab environments e.g., 64
        self.n_rollouts = n_rollouts    # minimum number of rollouts record for each command
        self.n_commands = n_commands    # discretization of the command space. Each axis gets n_commands bins
        self.command_names = [f"pos{i:03d}" for i in range(n_commands**3)]
        self.command_ranges = command_ranges    # dictionary of ranges that were to mdp.UniformPoseCommandCfg
        self.buffer = [[] for i in range(num_envs)] # buffer to store each environment's frames during individual rollouts
        self.counter = {k: int(0) for k in self.command_names}   # the number of complete, saved rollouts for each command
    
    def command_to_bin(self, command):
        x, y, z = command
        # Clamp to command space limits
        x = np.clip(x, self.command_ranges["xmin"], self.command_ranges["xmax"])
        y = np.clip(y, self.command_ranges["ymin"], self.command_ranges["ymax"])
        z = np.clip(z, self.command_ranges["zmin"], self.command_ranges["zmax"])
        # Normalize each coordinate to [0, 1]
        u = (x - self.command_ranges["xmin"]) / (self.command_ranges["xmax"] - self.command_ranges["xmin"])
        v = (y - self.command_ranges["ymin"]) / (self.command_ranges["ymax"] - self.command_ranges["ymin"])
        w = (z - self.command_ranges["zmin"]) / (self.command_ranges["zmax"] - self.command_ranges["zmin"])
        # Convert to bin indices in each dimension (0 ... n-1)
        ix = np.clip((u * self.n_commands).astype(np.int64), 0, self.n_commands-1)
        iy = np.clip((v * self.n_commands).astype(np.int64), 0, self.n_commands-1)
        iz = np.clip((w * self.n_commands).astype(np.int64), 0, self.n_commands-1)
        flat_index = ix + iy * self.n_commands + iz * (self.n_commands ** 2)
        return flat_index.item()
    
    def store_frames(self, frames, commands, dones):
        # store each frame into the buffer corresponding to its command. Flush the buffer if done.
        frames, commands, dones = self.preprocess(frames, commands, dones)  # convert everything to lists of size num_envs
        for i in range(self.num_envs):
            self.buffer[i] += [frames[i]]
            if dones[i]:
                command_name = self.command_names[self.command_to_bin(commands[i])]
                self.flush_frames(i, command_name)
        return self.check_if_done()
    
    def preprocess(self, frames, commands, dones):
        commands = commands[:,0:3].tolist()
        dones = dones.tolist()
        return frames, commands, dones
    
    def flush_frames(self, env_idx, command_name):
        # save the frames for a single rollout to disk
        rollout_name = f"{self.counter[command_name]:03d}"  # e.g., 000, 001, 002...
        path_base = os.path.join(self.root, command_name, rollout_name)
        os.makedirs(path_base)
        frames = self.buffer[env_idx]
        for i, frame in enumerate(frames):
            path = os.path.join(path_base, f"{i:03d}.png")
            img = Image.fromarray(frame)
            img.save(path)
        self.counter[command_name] += 1
        self.buffer[env_idx] = []
    
    def check_if_done(self):
        # returns True if all commands have the minimum number of rollouts
        status = [self.counter[k] for k in self.command_names]
        tf = min(status) >= self.n_rollouts
        return tf





@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # Load the most recent trained model, always
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    """Command names (e.g. pos000, pos001, pos002...) are the language conditions for the diffusion model.
    Each command name corresponds to a specific bin in the 3D command space."""

    n_commands = {"Lift-Cube-Randomized": 3,
                  "Lift-Cube-Deterministic": 3,
                  }

    # these should match the command configs in lift_env_cfg.py
    command_ranges = {"Lift-Cube-Randomized": {"xmin": 0.4, "xmax": 0.6,
                                               "ymin": -0.25, "ymax": 0.25,
                                               "zmin": 0.25, "zmax": 0.5,},
                      "Lift-Cube-Deterministic": {"xmin": 0.5, "xmax": 0.5,
                                                  "xmin": 0.0, "xmax": 0.0,
                                                  "xmin": 0.25, "xmax": 0.25,},
                      }
    
    # Custom dataset writer
    ds_writer = dataset_writer(root=os.path.join("./AVDC/datasets/isaaclab", args_cli.task_shorthand),
                               num_envs=args_cli.num_envs,
                               n_rollouts=args_cli.n_rollouts,
                               n_commands=n_commands[args_cli.task_shorthand],
                               command_ranges=command_ranges[args_cli.task_shorthand]
                               )

    dt = env.unwrapped.step_dt
    # reset environment
    obs = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            # thanks Grant
            frames = [isaac_utils.get_camera_frame(env.unwrapped, "overhead_camera", i)["rgb"] for i in range(args_cli.num_envs)]
            commands = env.unwrapped.command_manager.get_command("object_pose")
            if ds_writer.store_frames(frames, commands, dones):
                break
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()