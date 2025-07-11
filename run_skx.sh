#!/bin/bash

module load conda
conda activate pmoss

# Define model paths as variables
base_model="/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl"
assistant_model_amd_2s8n="/scratch/gilbreth/yrayhan/save_models/amd_epyc7543_2s_8n/0/2025-07-10-22-14-09-0.977.pkl"


# Example: inference for AMD 2s8n with ? model
# [11 12 44 45 16]
for wk in 13; do
  python run_dt_place.py \
    --mpath "$assistant_model_amd_2s8n" \
    --wl $wk \
    --ecfg 100 \
    --sidx 215 \
    --is_eval_only \
    --rtg 2
done

# For pre-training 
# python run_dt_place.py \
#   --mpath "None" \
#   --wl 12 \
#   --ecfg 100 \
#   --sidx 259 \
#   --rtg 2
