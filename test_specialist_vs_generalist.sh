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

# Define model paths as variables

# base_model_wo_ibm="None"
# python run_dt_place.py \
#   --mpath "$base_model_wo_ibm" \
#   --wl 11 \
#   --ecfg 100 \
#   --sidx 200 \
#   --rtg 2 \
#   --self_study \
#   --exclude_machine "amd_epyc7543_2s_8n"




base_model_wo_ibm_intel_skx_4s_8n="/scratch/gilbreth/yrayhan/save_models/self_study/intel_skx_4s_8n/2025-11-04-14-03-04-0.925.pkl"
base_model_wo_ibm_nvidia_gh_1s_1n="/scratch/gilbreth/yrayhan/save_models/self_study/nvidia_gh_1s_1n/2025-11-04-19-57-37-0.914.pkl"
base_model_wo_ibm_amd_epyc7543_2s_2n="/scratch/gilbreth/yrayhan/save_models/self_study/amd_epyc7543_2s_2n/2025-11-04-15-05-45-0.949.pkl"
base_model_wo_ibm_intel_sb_4s_4n="/scratch/gilbreth/yrayhan/save_models/self_study/intel_sb_4s_4n/2025-11-04-16-23-28-0.926.pkl"
base_model_wo_ibm_ibm_power9_2s_2n="None"

wl=11
nextCfg=60000
currCfg=100

python run_dt_place.py \
    --mpath "$base_model_wo_ibm_amd_epyc7543_2s_2n" \
    --wl "$wl" \
    --ecfg "$currCfg" \
    --sidx "$nextCfg" \
    --is_eval_only \
    --rtg 2

# base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/0/pretrain/2025-10-18-23-50-14-0.969.pkl"
# wl=11
# nextCfg=(60001)
# currCfg=100
# temperature=(1.6 1.7 1.8 1.9 1.10)
# # Loop through the arrays by using i
# for i in "${!nextCfg[@]}"; do
#   python run_dt_place.py \
#       --mpath "$base_model_wo_ibm" \
#       --wl "$wl" \
#       --ecfg "$currCfg" \
#       --sidx "${nextCfg[$i]}" \
#       --is_eval_only \
#       --rtg 2 
# done
    
# --temperature "${temperature[$i]}" \
    


