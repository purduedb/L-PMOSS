#!/bin/bash

module load conda
conda activate pmoss
base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/___d3rlpy_bc_models/2025-10-14-03-15-47-0.607.d3"
# ===========================================================================================
# For pre-training
python run_bc_v2.py \
  --mpath "$base_model_wo_ibm" \
  --wl 11 \
  --ecfg 100 \
  --sidx 200 \
  --rtg 2 \
  --epochs 400 \
  --n_embd 128 \
  --batch_size 256 \
  --save_path "/scratch/gilbreth/yrayhan/save_models/___d3rlpy_bc_models/"

# # ===========================================================================================
# # For inference on base models
# /scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/2025-10-11-06-46-38-0.917.d3: 1100s
# base_model_wo_ibm="/scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/2025-10-11-06-46-38-0.917.d3"
# wk_list=(11 12 16 44 45)
# sidx_list=(11000 11001 11002 11003 11004)

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

# # ===========================================================================================
# # For fine-tuning the model

# # python run_bc.py \
# #   --mpath "$base_model_wo_ibm" \
# #   --wl 11 \
# #   --ecfg 10000 \
# #   --sidx 10500 \
# #   --rtg 2 \
# #   --epochs 100 \
# #   --finetuning

# # ===========================================================================================
# # For inference on assistant models

# assistant_model_amd_epyc7543_2s_2n_v1="/scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/amd_epyc7543_2s_2n/2025-10-10-17-39-04-0.899.d3"
# # assistant_model_intel_skx_4s_8n_v1="/scratch/gilbreth/yrayhan/save_models/d3rlpy_bc_models/intel_skx_4s_8n/2025-10-10-17-03-38-0.900.d3"

# wk_list=(11 12 16 44 45)
# ecfg_list=(10000 10001 10002 10003 10004)
# sidx_list=(10500 10501 10502 10503 10504)

# for i in "${!wk_list[@]}"; do
#   wk=${wk_list[$i]}
#   ecfg=${ecfg_list[$i]}
#   sidx=${sidx_list[$i]}

#   python run_bc.py \
#     --mpath "$assistant_model_intel_skx_4s_8n_v1" \
#     --wl "$wk" \
#     --ecfg "$ecfg" \
#     --sidx "$sidx" \
#     --is_eval_only \
#     --rtg 2 
# done
