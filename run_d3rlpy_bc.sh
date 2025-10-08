#!/bin/bash

module load conda
conda activate pmoss


# For pre-training 
# python run_bc.py \
#   --mpath "None" \
#   --wl 11 \
#   --ecfg 100 \
#   --sidx 200 \
#   --rtg 2 

base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/2025-10-08-12-17-16-0.160.d3"
# # Traditional inference for seen workloads
wk_list=(11)
sidx_list=(200000)

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

