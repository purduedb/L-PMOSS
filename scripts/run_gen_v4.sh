#!/bin/bash

module load conda
conda activate pmoss

# Define model paths as variables
# The default base model with default DT setting
current_base_model_wo_ibm="None"

python run_dt_place_v3.py \
  --mpath "$current_base_model_wo_ibm" \
  --wl 11 --ecfg 100 --sidx 200 --rtg 2 \
  --generalization_study \
  --exclude_machine "intel_sb_4s_4n"
