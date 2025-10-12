#!/bin/bash

module load conda
conda activate pmoss

base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_cql_models/2025-10-11-08-06-28-0.385.d3"
# For pre-training 
python run_cql.py \
  --mpath "None" \
  --wl 11 \
  --ecfg 100 \
  --sidx 200 \
  --rtg 2 \
  --n_embd 512 \
  --epochs 400

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

