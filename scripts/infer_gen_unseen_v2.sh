#!/bin/bash

module load conda
conda activate pmoss

base_model_wo_intel_skx_4s_8n=/scratch/gilbreth/yrayhan/save_models/base_models/intel_skx_4s_8n/0/2025-10-06-04-28-16-0.917.pkl
base_model_wo_intel_sb_4s_4n=/scratch/gilbreth/yrayhan/save_models/base_models/intel_sb_4s_4n/0/2025-10-06-05-10-35-0.917.pkl
base_model_wo_amd_epyc7543_2s_8n=/scratch/gilbreth/yrayhan/save_models/base_models/amd_epyc7543_2s_8n/0/2025-10-06-04-40-58-0.909.pkl
base_model_wo_amd_epyc7543_2s_2n=/scratch/gilbreth/yrayhan/save_models/base_models/amd_epyc7543_2s_2n/0/2025-10-06-04-18-50-0.910.pkl


# Declare wkload array 
workloads=(11 12)
# Revisit the sidx_list
sidx_list=(2000 2001)
for i in "${!workloads[@]}"; do
  wl=${workloads[$i]}
  sidx=${sidx_list[$i]}
  
  python run_dt_place_v2.py \
    --mpath "$base_model_wo_amd_epyc7543_2s_2n" \
    --wl "$wl" \
    --ecfg 100 \
    --sidx "$sidx" \
    --is_eval_only \
    --rtg 2 \
    --generalization_study \
    --exclude_machine "amd_epyc7543_2s_2n" 
done


