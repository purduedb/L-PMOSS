#!/bin/bash

# Create or attach to tmux session for persistent SLURM interactive jobs
SESSION_NAME="dt_inference"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  # Create new session in detached mode
  tmux new-session -d -s $SESSION_NAME
  echo "Created new tmux session: $SESSION_NAME"
  
  # Load conda in the session
  tmux send-keys -t $SESSION_NAME "module load conda" C-m
  tmux send-keys -t $SESSION_NAME "conda activate pmoss" C-m
  sleep 1
  
  # Navigate to working directory
  tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
  
  echo ""
  echo "==================================================================="
  echo "Tmux session created! Now attaching you to it..."
  echo "==================================================================="
  sleep 1
  
  # Auto-attach to the session
  tmux attach -t $SESSION_NAME
else
  echo "Session $SESSION_NAME already exists. Attaching..."
  tmux attach -t $SESSION_NAME
fi
