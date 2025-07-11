#!/bin/bash

module load conda
conda activate pmoss

# Define model paths as variables
base_model_wo_nvidia_ibm="/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl"
base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/0/2025-07-11-09-24-15-0.923.pkl"
assistant_model_amd_2s8n_v1="/scratch/gilbreth/yrayhan/save_models/amd_epyc7543_2s_8n/0/2025-07-10-22-14-09-0.977.pkl"
assistant_model_amd_2s8n_v2="/scratch/gilbreth/yrayhan/save_models/amd_epyc7543_2s_8n/0/2025-07-11-00-39-15-0.979.pkl"

# Example: inference for AMD 2s8n with ? model
# [44 ]
# for wk in 16; do
#   python run_dt_place.py \
#     --mpath "$assistant_model_amd_2s8n_v2" \
#     --wl $wk \
#     --ecfg 100 \
#     --sidx 221 \
#     --is_eval_only \
#     --rtg 2
# done

# For pre-training 
python run_dt_place.py \
  --mpath "$base_model_wo_ibm" \
  --wl 12 \
  --ecfg 100 \
  --sidx 259 \
  --rtg 2
