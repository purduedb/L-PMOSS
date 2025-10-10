#!/bin/bash

module load conda
conda activate pmoss

base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_cql_models/2025-10-10-08-58-07-0.188.d3"
# For pre-training 
python run_cql.py \
  --mpath "$base_model_wo_ibm" \
  --wl 11 \
  --ecfg 100 \
  --sidx 200 \
  --rtg 2 

# base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_cql_models/2025-10-08-14-14-34-0.016.d3"

# wk_list=(11)
# sidx_list=(300000)

# for i in "${!wk_list[@]}"; do
#   wk=${wk_list[$i]}
#   sidx=${sidx_list[$i]}

#   python run_cql.py \
#     --mpath "$base_model_wo_ibm" \
#     --wl "$wk" \
#     --ecfg 100 \
#     --sidx "$sidx" \
#     --is_eval_only \
#     --rtg 2 
# done

