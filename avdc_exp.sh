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
    --video_folder "results/franka_test_selected" \
    --video_ckpt_path "AVDC/results/mw/model-24.pt" \
    > "debug.txt"
done

