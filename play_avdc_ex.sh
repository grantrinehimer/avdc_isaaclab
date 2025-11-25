python scripts/rsl_rl/play_avdc.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --video_ckpt_dir AVDC/results/mw \
  --video_milestone 24 \
  --task_prompt "lift_custom" \
  --plan_timeout 15 \
  --camera_sensor overhead_camera \
  --headless > debug/log