#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

mkdir -p debug

for seed in $(seq 1 10); do
  echo "Running AVDC experiment with random seed ${seed}"
  python scripts/rsl_rl/play_avdc.py \
    --device=cpu \
    --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
    --video_ckpt_dir AVDC/results/mw \
    --video_milestone 24 \
    --task_prompt "lift_custom" \
    --plan_timeout 15 \
    --camera_sensor overhead_camera \
    --save_video \
    --video_length 2000 \
    --video_folder "results/point_tracker/franka_test_sampled" \
    --random_seed ${seed} \
    --diffusion_source images \
    --sample_images True \
    --motion_backend flow \
    --locotrack_root "/home/grant-rinehimer/cornell/dl_robotics/avdc_isaaclab/locotrack/locotrack_pytorch" \
    --locotrack_ckpt "/home/grant-rinehimer/cornell/dl_robotics/avdc_isaaclab/locotrack/locotrack_pytorch/ckpt/locotrack_base.ckpt" \
    --locotrack_model_size base \
    --locotrack_query_chunk_size 256 \
    --locotrack_max_points 256 \
    --locotrack_sample_points 512 \
    --locotrack_min_points 32 \
    > "debug_${seed}.txt"
done

