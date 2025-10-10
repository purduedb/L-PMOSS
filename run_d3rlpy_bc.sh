#!/bin/bash

module load conda
conda activate pmoss

base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/2025-10-09-21-58-23-0.568.d3"

# For pre-training 
python run_bc.py \
  --mpath "None" \
  --wl 11 \
  --ecfg 100 \
  --sidx 200 \
  --rtg 2 \
  --epochs 400

# base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/2025-10-08-19-56-29-0.064.d3"

# # Traditional inference for seen workloads
# wk_list=(11)
# sidx_list=(200000)

# for i in "${!wk_list[@]}"; do
#   wk=${wk_list[$i]}
#   sidx=${sidx_list[$i]}

#   python run_bc.py \
#     --mpath "$base_model_wo_ibm" \
#     --wl "$wk" \
#     --ecfg 100 \
#     --sidx "$sidx" \
#     --is_eval_only \
#     --rtg 2 
# done

