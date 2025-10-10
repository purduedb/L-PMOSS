#!/bin/bash

# All the IBM model's inference experiments for gen are here 
module load conda
conda activate pmoss

base_model_wo_intel_skx_4s_8n=/scratch/gilbreth/yrayhan/save_models/base_models/intel_skx_4s_8n/0/2025-10-06-04-28-16-0.917.pkl
base_model_wo_intel_sb_4s_4n=/scratch/gilbreth/yrayhan/save_models/base_models/intel_sb_4s_4n/0/2025-10-06-05-10-35-0.917.pkl
base_model_wo_amd_epyc7543_2s_8n=/scratch/gilbreth/yrayhan/save_models/base_models/amd_epyc7543_2s_8n/0/2025-10-06-04-40-58-0.909.pkl
base_model_wo_amd_epyc7543_2s_2n=/scratch/gilbreth/yrayhan/save_models/base_models/amd_epyc7543_2s_2n/0/2025-10-10-02-15-51-0.916.pkl

# For wkload 11: [2025-10-10-02-15-51-0.916.pkl: 2001], 910=2000, 935=2000, 2025-10-10-02-09-40-0.896.pkl:2002, 2025-10-10-01-37-00-0.868.pkl: 2003
# 2025-10-10-03-42-52-0.917.pkl: 2004, 2025-10-10-05-12-22-0.946.pkl: 2005, 2025-10-06-03-11-21-0.810.pkl: 2006
# 2025-10-10-06-07-34-0.909.pkl: 2007, 2025-10-10-06-19-58-0.931.pkl: 2008, 2025-10-10-07-35-21-0.953.pkl: 2009
workloads=(12)
sidx_list=(2010)
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


