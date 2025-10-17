#!/bin/bash

module load conda
conda activate pmoss

initial_load="/scratch/gilbreth/yrayhan/save_models/___d3rlpy_cql_models/2025-10-15-18-12-23-0.875.d3" 
# For pre-training 
python run_cql_v2.py \
  --mpath "$initial_load" \
  --wl 11 \
  --ecfg 100 \
  --sidx 200 \
  --rtg 2 \
  --n_embd 32 \
  --epochs 800 \
  --save_path "/scratch/gilbreth/yrayhan/save_models/___d3rlpy_cql_models/"

# base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_cql_models/2025-10-12-14-00-54-0.892.d3"

# wk_list=(11 12 45 16)
# sidx_list=(12000 12001 12003 12004)

# for i in "${!wk_list[@]}"; do
#   wk=${wk_list[$i]}
#   sidx=${sidx_list[$i]}

#   python run_cql_v2.py \
#     --mpath "$base_model_wo_ibm" \
#     --wl "$wk" \
#     --ecfg 100 \
#     --sidx "$sidx" \
#     --is_eval_only \
#     --rtg 2 
# done

