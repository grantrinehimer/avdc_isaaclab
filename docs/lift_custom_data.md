## Some elements of this are wrong, ignore for now

# Lift Custom Data Pipeline

This document explains how to configure the new overhead camera, obtain an expert PPO policy, and record Isaac Lab lift demonstrations that match the `SequentialDatasetv2` layout used by `AVDC/flowdiffusion/train_mw.py`.

---

## 1. Camera configuration

- The lift-specific environment now owns its own copy of `LiftEnvCfg` at `source/avdc_isaaclab/avdc_isaaclab/tasks/manager_based/lift_custom/lift_env_cfg.py`.
- `ObjectTableSceneCfg.overhead_camera` defines a world-frame RGB camera aimed at the cube workspace. By default it sits at `(0.55, 0.0, 0.85)` looking straight down.
- Resolution is controlled via `LiftEnvCfg.camera_resolution` (default `320x240`). Any task that sets this tuple will automatically propagate the change to the USD camera spawn configuration.
- To adjust the viewpoint, edit the offset:
  ```python
  overhead_camera = CameraCfg(
      ...
      offset=CameraCfg.OffsetCfg(
          pos=(0.55, 0.0, 0.85),
          rot=(0.0, 1.0, 0.0, 0.0),  # 180° pitch in ROS convention for a top-down view
          convention="ros",
      ),
  )
  ```
  Move `pos` to change the mount location or supply a quaternion that looks along a different direction (e.g., a front oblique shot).
- Remember to launch Isaac Sim with `--enable_cameras`; the collector enforces this automatically, but it is also required if you want to watch the scene live.

---

## 2. PPO expert training (or reuse)

1. **Training command**  
   Run the standard RSL-RL trainer with the relative IK task so the camera-equipped environment is exercised:
   ```bash
   ./isaaclab.sh -p scripts/rsl_rl/train.py \
     --task Isaac-Lift-Cube-Franka-Custom-IK-Rel-v0 \
     --agent rsl_rl_cfg_entry_point \
     --num_envs 4096 \
     --enable_cameras \
     --experiment_name franka_lift
   ```
   - `LiftEnvCfg.camera_resolution` and the new camera live inside the task config, so no extra Hydra overrides are required.
   - Training artefacts are stored under `logs/rsl_rl/franka_lift/<timestamp_run>/`.

2. **Automatic fallback expert**  
   The recorder uses `avdc_isaaclab.tasks.manager_based.lift_custom.expert.resolve_expert_checkpoint()` to locate a checkpoint:
   - It first searches the local `logs/rsl_rl/<experiment_name>` directory (respecting `--load_run` / `--load_checkpoint` overrides).
   - If nothing is found, it downloads NVIDIA’s published `franka_lift` PPO weights via `isaaclab.utils.pretrained_checkpoint`.
   - You can still supply `--checkpoint /path/to/policy.pt` to override both behaviours.

This means you can either train your own expert for the modified environment or immediately leverage the published one for data generation.

---

## 3. Recording expert rollouts

Use the dedicated collector to save RGB frames and actions in the structure expected by `SequentialDatasetv2`:

```bash
python scripts/collect_lift_rollouts.py \
  --task Isaac-Lift-Cube-Franka-Custom-IK-Rel-v0 \
  --num_rollouts 10 \
  --max_steps 250 \
  --dataset_root AVDC/datasets/isaaclab \
  --dataset_task lift_custom \
  --dataset_variant overhead \
  --camera_width 320 \
  --camera_height 240
```

- Key CLI options:
  - `--camera_width/--camera_height`: override the capture resolution without touching the config file.
  - `--start_index`: continue numbering rollouts if you pause/resume collection.
  - `--camera_sensor`: switch to a different sensor name if you add more cameras later.
  - `--metadata`: store a `metadata.json` alongside each trajectory with the checkpoint path and success flag.
- Output layout (mirrors Metaworld data):
  ```
  <dataset_root>/metaworld_dataset/<task>/<variant>/<episode_id>/
      00.png
      01.png
      ...
      action.pkl          # dict with actions, num_steps and success flag
      metadata.json       # (optional) camera + checkpoint metadata
  ```
- Each episode contains the initial frame (`00.png`) before any action and then one frame per environment step, so `SequentialDatasetv2` can treat the first frame as the conditioning image.

You can run the collector repeatedly to accumulate thousands of trajectories; it will auto-increment the episode folder and skip existing IDs.

---

## 4. Training the video diffusion model

Point the dataset root in `train_mw.py` (or any other consumer) to the lift dataset:

```python
train_set = SequentialDatasetv2(
    sample_per_seq=8,
    path="AVDC/datasets/isaaclab",  # root containing metaworld_dataset/...
    target_size=(128, 128),
    randomcrop=True,
)
```

Because the directory hierarchy and PNG naming follow the existing Metaworld datasets, no further code changes are needed—`SequentialDatasetv2` will discover the `lift_custom/overhead/<episode>` folders automatically.

---

## Quick checklist

1. **Update camera & resolution** (optional)  
   Edit `LiftEnvCfg.camera_resolution` or the `CameraCfg.OffsetCfg`.
2. **Train or reuse PPO expert**  
   `scripts/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-Custom-IK-Rel-v0 ...`  
   (or rely on the published checkpoint).
3. **Collect rollouts**  
   `python scripts/collect_lift_rollouts.py --num_rollouts 100 ...`
4. **Train diffusion policy**  
   Point `SequentialDatasetv2.path` at your newly recorded dataset.

This pipeline produces camera-aligned, expert-labelled demonstrations that can be fed directly into `AVDC/flowdiffusion/train_mw.py` and later consumed by `MyPolicy_CL`.

