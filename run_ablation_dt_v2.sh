#!/bin/bash

module load conda
conda activate pmoss

# Define model paths as variables
# The default base model with default DT setting
default_base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/0/2025-07-11-12-17-34-0.912.pkl"
current_base_model_wo_ibm="None"

num_layer=(4) # default is 6
num_heads=(2 4 6)
embed_size=(16 32 64)


for i in "${!num_layer[@]}"; do
  nl=${num_layer[$i]}
  # nh=${num_heads[$i]}
  # es=${embed_size[$i]}
  python run_dt_place.py \
    --mpath "$current_base_model_wo_ibm" \
    --wl 11 \
    --ecfg 100 \
    --sidx 200 \
    --rtg 2 \
    --ablation_study \
    --ablation_param num_layer \
    --n_layer "$nl"
done

