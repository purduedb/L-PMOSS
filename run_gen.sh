#!/bin/bash

module load conda
conda activate pmoss

# Define model paths as variables
# The default base model with default DT setting
current_base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/base_models/intel_skx_4s_8n/0/2025-10-05-22-47-25-0.350.pkl"


python run_dt_place.py \
  --mpath "$current_base_model_wo_ibm" \
  --wl 11 --ecfg 100 --sidx 200 --rtg 2 \
  --generalization_study \
  --exclude_machine "intel_skx_4s_8n"
