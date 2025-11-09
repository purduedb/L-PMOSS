#!/bin/bash

module load conda
conda activate pmoss

# # Checkout to make sure it lands on the correct branch
# expected_branch="foundational-rcac-v2"
# current_branch=$(git branch --show-current)

# if [ "$current_branch" != "$expected_branch" ]; then
#     echo "Switching from '$current_branch' to '$expected_branch'..."
#     git checkout "$expected_branch"
#     if [ $? -ne 0 ]; then
#         echo "Error: Failed to switch to branch '$expected_branch'"
#         exit 1
#     fi
# else
#     echo "Already on branch '$expected_branch'"
# fi


# Pre-trained model
# base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/0/pretrain/2025-10-18-23-50-14-0.969.pkl"


# # Post-training
# python run_dt_place.py \
#   --mpath "$base_model_wo_ibm" \
#   --wl 11 \
#   --ecfg 100 \
#   --sidx 200 \
#   --rtg 2 \
#   --post_train \

# infer on post-trained models
post_train_intel_skx_4s_8n="/scratch/gilbreth/yrayhan/save_models/post_train/intel_skx_4s_8n/2025-11-09-17-46-17-0.964.pkl"
post_train_nvidia_gh_1s_1n="/scratch/gilbreth/yrayhan/save_models/post_train/nvidia_gh_1s_1n/2025-11-09-18-00-26-0.961.pkl"
post_train_amd_epyc7543_2s_2n="/scratch/gilbreth/yrayhan/save_models/post_train/amd_epyc7543_2s_2n/2025-11-09-17-53-48-0.955.pkl"

wl=11
nextCfg=70000
currCfg=100

python run_dt_place.py \
    --mpath "$post_train_nvidia_gh_1s_1n" \
    --wl "$wl" \
    --ecfg "$currCfg" \
    --sidx "$nextCfg" \
    --is_eval_only \
    --rtg 2