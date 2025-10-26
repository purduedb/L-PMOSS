#!/bin/bash

module load conda
conda activate pmoss

# Checkout to make sure it lands on the correct branch
expected_branch="foundational-rcac-v2"
current_branch=$(git branch --show-current)

if [ "$current_branch" != "$expected_branch" ]; then
    echo "Switching from '$current_branch' to '$expected_branch'..."
    git checkout "$expected_branch"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to switch to branch '$expected_branch'"
        exit 1
    fi
else
    echo "Already on branch '$expected_branch'"
fi

# Define model paths as variables
base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/0/pretrain/2025-10-18-23-50-14-0.969.pkl"

# For inference
# They are inputs from command line arguments

wl=$1
nextCfg=$2
currCfg=$3

python run_dt_place.py \
    --mpath "$base_model_wo_ibm" \
    --wl "$wl" \
    --ecfg "$currCfg" \
    --sidx "$nextCfg" \
    --is_eval_only \
    --rtg 2



