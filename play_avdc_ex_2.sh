python scripts/rsl_rl/play_avdc.py \
  --device=cpu \
  --headless \
  --task Isaac-Lift-Cube-UR10-IK-Rel-v0 \
  --video_ckpt_dir "AVDC/results/isaaclab/Lift-Cube-Randomized" \
  --video_milestone 24 \
  --task_prompt pos010 \
  --save_video \
  --diffusion_source model
  > debug/log