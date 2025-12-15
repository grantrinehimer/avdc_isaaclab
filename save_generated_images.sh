#!/bin/bash

# Define the start and end position indices
START_POS=7
END_POS=25

# Define the common script arguments
HEADLESS="--headless"
TASK="--task Isaac-Lift-Cube-Franka-IK-Rel-v0"
VIDEO_MILESTONE="--video_milestone 24"
VIDEO_CKPT_PATH=AVDC/results/isaaclab/Lift-Cube-Randomized/model-24.pt
SAVE_VIDEO="--save_video"
VIDEO_FOLDER=videos
DIFFUSION_SOURCE="--diffusion_source model"
PYTHON_SCRIPT="python scripts/rsl_rl/play_avdc.py"
RESULTS_CSV=results.csv
SAVE_GENERATED="--save_generated"

# Ensure the debug directory exists for the logs
LOG_DIR="debug"
mkdir -p "$LOG_DIR"

echo "Starting script execution from pos${START_POS} to pos${END_POS}..."

# Loop from START_POS to END_POS (inclusive)
for i in $(seq $START_POS $END_POS); do
    # Format the position number with leading zeros (e.g., 0 -> 000, 26 -> 026)
    POS_NUM=$(printf "%03d" $i)
    TASK_PROMPT="--task_prompt \"pos${POS_NUM}\""
    
    # Construct the log file name: log_pos000, log_pos001, etc.
    LOG_FILE="${LOG_DIR}/log_pos${POS_NUM}"

    echo "--- Executing for pos${POS_NUM} (Logging to ${LOG_FILE}) ---"

    # Execute the command
    # The variables are intentionally left unquoted here ($PYTHON_SCRIPT ...) 
    # because the arguments within them already contain necessary quotes (e.g., "$VIDEO_CKPT_PATH").
    $PYTHON_SCRIPT \
        $SAVE_GENERATED \
        $HEADLESS \
        $TASK \
        $VIDEO_MILESTONE \
        --video_ckpt_path "$VIDEO_CKPT_PATH" \
        --task_prompt "pos${POS_NUM}" \
        $SAVE_VIDEO \
        --video_folder "$VIDEO_FOLDER" \
        $DIFFUSION_SOURCE \
        --results_csv "$RESULTS_CSV" \
        > "$LOG_FILE" 2>&1 # Redirect stdout (>) and stderr (2>&1) to the log file

    # Check the exit status of the previous command
    if [ $? -eq 0 ]; then
        echo "Successfully completed pos${POS_NUM}."
    else
        echo "An error occurred during execution for pos${POS_NUM}. Check ${LOG_FILE} for details."
    fi
    echo "" # Add a newline for readability in the main terminal output
done

echo "Script execution complete. Logs are available in the '${LOG_DIR}' directory."