#!/bin/bash

module load conda
conda activate pmoss

# Define model paths as variables
base_model_wo_nvidia_ibm="/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl"
base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/0/2025-07-11-12-17-34-0.912.pkl"
base_model_w_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/0/2025-07-11-16-10-07-0.820.pkl"

assistant_model_amd_2s8n_v1="/scratch/gilbreth/yrayhan/save_models/amd_epyc7543_2s_8n/0/2025-07-10-22-14-09-0.977.pkl"
assistant_model_amd_2s8n_v2="/scratch/gilbreth/yrayhan/save_models/amd_epyc7543_2s_8n/0/2025-07-11-00-39-15-0.979.pkl"

assistant_model_intel_sb_4s_4n_v1="/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-11-17-09-47-0.985.pkl"

# Example: inference for AMD 2s8n with ? model
# Intel SB: 4s4n []
# 11:200 12: 201, 44: 202, 45: 203, 16: 204, 13: 205
# [11 12]
for wk in 11; do
  python run_dt_place.py \
    --mpath "$assistant_model_intel_sb_4s_4n_v1" \
    --wl $wk \
    --ecfg 100 \
    --sidx 210 \
    --is_eval_only \
    --rtg 2
done


# For pre-training 
# python run_dt_place.py \
#   --mpath "$assistant_model_intel_sb_4s_4n_v1" \
#   --wl 11 \
#   --ecfg 100 \
#   --sidx 200 \
#   --rtg 2


