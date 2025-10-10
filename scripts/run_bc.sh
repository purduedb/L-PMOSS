#!/bin/bash

module load conda
conda activate pmoss

base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/bc_models/0/2025-10-07-00-23-22-0.917.pkl"

# For pre-training 
# python run_dt_place.py \
#   --mpath "None" \
#   --wl 11 \
#   --ecfg 100 \
#   --sidx 200 \
#   --rtg 2 \
#   --model_type naive


# Traditional inference for seen workloads
wk_list=(12 )
sidx_list=(5001)

for i in "${!wk_list[@]}"; do
  wk=${wk_list[$i]}
  sidx=${sidx_list[$i]}

  python run_dt_place.py \
    --mpath "$base_model_wo_ibm" \
    --wl "$wk" \
    --ecfg 100 \
    --sidx "$sidx" \
    --is_eval_only \
    --rtg 2 \
    --model_type naive
done

# wk_list=(11 12 16 44 45 13)
# sidx_list=(5000 5001 5002 5003 5004 5005)

# for i in "${!wk_list[@]}"; do
#   wk=${wk_list[$i]}
#   sidx=${sidx_list[$i]}

#   python run_dt_place.py \
#     --mpath "$base_model_wo_ibm" \
#     --wl "$wk" \
#     --ecfg 100 \
#     --sidx "$sidx" \
#     --is_eval_only \
#     --rtg 2 \
#     --model_type naive
# done




