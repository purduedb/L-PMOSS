#!/bin/bash

module load conda
conda activate pmoss

# ===========================================================================================
# For pre-training
python run_bc.py \
  --mpath "None" \
  --wl 11 \
  --ecfg 100 \
  --sidx 200 \
  --rtg 2 \
  --epochs 400 \
  --n_embd 128 \
  --batch_size 256 \
  --save_path "/scratch/gilbreth/xxxxxxx/save_models/__d3rlpy_bc_models/"

# # ===========================================================================================

base_model_wo_ibm="/scratch/gilbreth/xxxxxxx/save_models/__d3rlpy_bc_models/2025-10-14-12-31-03-0.711.d3"
wk_list=(11 12 16 44 45)
sidx_list=(14000 14001 14004 14002 14003)

for i in "${!wk_list[@]}"; do
  wk=${wk_list[$i]}
  sidx=${sidx_list[$i]}

  python run_bc.py \
    --mpath "$base_model_wo_ibm" \
    --wl "$wk" \
    --ecfg 100 \
    --sidx "$sidx" \
    --is_eval_only \
    --rtg 2 \
    --n_embd 128 \
    --batch_size 256
done
