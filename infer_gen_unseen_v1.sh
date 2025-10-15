#!/bin/bash

module load conda
conda activate pmoss

# 2000s: 2025-10-06-04-28-16-0.917.pkl, 2025-10-12-05-37-27-0.946.pkl: 3000s
base_model_wo_intel_skx_4s_8n=/scratch/gilbreth/yrayhan/save_models/base_models/intel_skx_4s_8n/0/2025-10-12-05-37-27-0.946.pkl
# 2000s:2025-10-06-05-10-35-0.917.pkl 
base_model_wo_intel_sb_4s_4n=/scratch/gilbreth/yrayhan/save_models/base_models/intel_sb_4s_4n/0/2025-10-12-11-24-09-0.937.pkl
# 2025-10-06-04-40-58-0.909.pkl: 2000s, 2025-10-12-19-31-09-0.933.pkl: 3000s
base_model_wo_amd_epyc7543_2s_8n=/scratch/gilbreth/yrayhan/save_models/base_models/amd_epyc7543_2s_8n/0/2025-10-12-19-31-09-0.933.pkl
# 2025-10-12-00-54-08-0.903.pkl:3000, 2025-10-12-01-57-15-0.926.pkl:3010
base_model_wo_amd_epyc7543_2s_2n=/scratch/gilbreth/yrayhan/save_models/base_models/amd_epyc7543_2s_2n/0/2025-10-12-00-54-08-0.903.pkl

base_model_wo_nvidia_gh_1s_1n=/scratch/gilbreth/yrayhan/save_models/base_models/nvidia_gh_1s_1n/0/2025-10-12-11-20-19-0.960.pkl


# Declare wkload array 
workloads=(11 12 45 16)
# Revisit the sidx_list
sidx_list=(3000 3001 3003 3004)
for i in "${!workloads[@]}"; do
  wl=${workloads[$i]}
  sidx=${sidx_list[$i]}
  
  python infer_run_dt_place.py \
    --mpath "$base_model_wo_amd_epyc7543_2s_8n" \
    --wl "$wl" \
    --ecfg 100 \
    --sidx "$sidx" \
    --is_eval_only \
    --rtg 2 \
    --generalization_study \
    --exclude_machine "amd_epyc7543_2s_8n" 
done


