# MyPolicy_CL IO expectations

Quick reference for anyone wiring `MyPolicy_CL` into a new simulator (e.g.
IsaacLab). The goal is to keep the diffusion + optical-flow stack untouched by
feeding it the same data it expects from Meta-World.

## EnvAdapter contract

- Instantiate `MyPolicy_CL(..., env_adapter=adapter)` where `adapter` implements
  the methods defined in `env_interfaces/base.py`.
- The adapter is responsible for `reset/step` passthrough, plus RGB-D frames,
  segmentation masks, and camera intrinsics via `fetch_*` methods.
- For legacy Meta-World runs, `MetaworldEnvAdapter` (in
  `metaworld_exp/utils.py`) wraps Mujoco envs without code changes.

## Perception

- RGB frame: `uint8` array shaped `(H, W, 3)` (default `(240, 320, 3)`).
- Depth map: float array shaped `(H, W)` aligned with the RGB frame (meters).
- Segmentation mask: single-channel mask (bool or `uint8` `{0,255}`) highlighting
  the manipulated object(s). Passed directly to `get_transforms`.
- Camera matrix: `3x4` projection matrix produced by `get_cmat` in the original
  utilities (`image @ focal @ rotation @ translation`). The planner uses it to
  lift 2-D samples into 3-D before fitting rigid transforms.

Only the first RGB-D snapshot comes from the simulator. `pred_video` handles the
future frames internally, and `pred_flow_frame` takes those frames to build the
flow stack consumed by `get_transforms`.

## Manipulation

- Observation parsing assumes the first 3 floats are the end-effector position
  (world frame). Nothing else from the observation vector is used.
- Actions are a 4-D vector: `Δx, Δy, Δz` (Cartesian deltas from `move(...)`)
  plus a scalar `grab_effort` in `{-0.8, 0.8}`.
- Mode-dependent behavior:
  - `grasp`: lift to `grasp + [0, 0, 0.2]`, descend, close, then execute
    sub-goals generated from rigid transforms.
  - `push`: shift grasp points by `-0.03 m` in Z and offset the initial pose by
    `0.08 m` opposite the first sub-goal for stability.
- Replanning: if the hand stalls for `plan_timeout` steps or finishes all
  sub-goals, `calculate_next_plan()` is called again, so adapters must be able to
  fetch fresh RGB-D/mask/camera matrix any time.

As long as a new environment supplies these exact signals (possibly through a
small adapter), the policy can stay unchanged while teammates update the shared
utilities for IsaacSim.

