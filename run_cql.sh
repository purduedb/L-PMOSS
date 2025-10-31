#!/bin/bash

module load conda
conda activate pmoss

# # For pre-training 
python run_cql.py \
  --mpath "$initial_load" \
  --wl 11 \
  --ecfg 100 \
  --sidx 200 \
  --rtg 2 \
  --n_embd 128 \
  --epochs 800 \
  --save_path "/scratch/gilbreth/xxxxxxx/save_models/-d3rlpy_cql_models/"

base_model_wo_ibm="/scratch/gilbreth/xxxxxxx/save_models/-d3rlpy_cql_models/2025-10-17-07-09-52-0.800.d3"

wk_list=(11 12 44 45 16)
sidx_list=(15000 15001 15002 15003 15004)

for i in "${!wk_list[@]}"; do
  wk=${wk_list[$i]}
  sidx=${sidx_list[$i]}

  python run_cql.py \
  --mpath "$base_model_wo_ibm" \
  --wl "$wk" \
  --ecfg 100 \
  --sidx "$sidx" \
  --rtg 2 \
  --n_embd 128 \
  --epochs 800 \
  --is_eval_only
done

