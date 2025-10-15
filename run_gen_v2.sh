#!/bin/bash

module load conda
conda activate pmoss

# Define model paths as variables
# The default base model with default DT setting
# current_base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/amd_epyc7543_2s_2n/0/2025-10-10-05-12-22-0.946.pkl"

# Task: prepare stuff for nvidia 

python run_dt_place_v2.py \
  --mpath "None" \
  --wl 11 --ecfg 100 --sidx 200 --rtg 2 \
  --generalization_study \
  --exclude_machine "intel_sb_4s_4n" \
