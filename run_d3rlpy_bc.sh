#!/bin/bash

module load conda
conda activate pmoss

base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/2025-10-10-12-40-04-0.896.d3"

# For pre-training 
# python run_bc.py \
#   --mpath "$base_model_wo_ibm" \
#   --wl 11 \
#   --ecfg 100 \
#   --sidx 200 \
#   --rtg 2 \
#   --epochs 400


# For inference on base models
wk_list=(11 12 16 44 45)
sidx_list=(10000 10001 10002 10003 10004)

for i in "${!wk_list[@]}"; do
  wk=${wk_list[$i]}
  sidx=${sidx_list[$i]}

  python run_bc.py \
    --mpath "$base_model_wo_ibm" \
    --wl "$wk" \
    --ecfg 100 \
    --sidx "$sidx" \
    --is_eval_only \
    --rtg 2 
done

