#!/bin/bash

#!/bin/bash

# python run_dt_place.py --wl 12 --ecfg 36 --sidx 204
# python run_dt_place.py --wl 12 --ecfg 20 --sidx 205


# H11
# python run_dt_place.py --wl 32 --ecfg 102 --sidx 201
# python run_dt_place.py --wl 32 --ecfg 100 --sidx 203


# python run_dt_place.py --wl 12 --ecfg 102 --sidx 201
# python run_dt_place.py --wl 12 --ecfg 20 --sidx 205
# python run_dt_place.py --wl 13 --ecfg 59 --sidx 205
# python run_dt_place.py --wl 32 --ecfg 100 --sidx 203
# python run_dt_place.py --wl 32 --ecfg 29 --sidx 202

# without any provisioning of hthe masks 
# WKLOAD E
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 100 --sidx 206
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 101 --sidx 207
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 34 --sidx 208
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 62 --sidx 209

#WKLOAD 
conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 100 --sidx 204
conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 101 --sidx 205
conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 102 --sidx 206
conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 29 --sidx 207
# Sucks big time


2024-10-22-18-22-39-0.872 Model 
# Sucks big time
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 100 --sidx 207
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 101 --sidx 208
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 34 --sidx 209
conda activate idx_creator
python run_dt_place.py --wl 13 --ecfg 62 --sidx 210

conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 100 --sidx 208
conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 101 --sidx 209
conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 102 --sidx 210
conda activate idx_creator
python run_dt_place.py --wl 32 --ecfg 29 --sidx 211


# 2024-10-22-21-11-38-0.929
# start from 212

conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 1.1 --ecfg 100 --sidx 220
conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 1.1 --ecfg 101 --sidx 221
conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 1.1 --ecfg 27 --sidx 222
conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 1.1 --ecfg 36 --sidx 223

conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 2 --ecfg 100 --sidx 224
conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 2 --ecfg 101 --sidx 225
conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 2 --ecfg 27 --sidx 226
conda activate idx_creator
python run_dt_place.py --wl 12 --rtg 2 --ecfg 36 --sidx 2227



# conda activate idx_creator
# python run_dt_place.py --wl 39 --rtg 1.1 --ecfg 100 --sidx 200
# conda activate idx_creator
# python run_dt_place.py --wl 39 --rtg 1.1 --ecfg 101 --sidx 201
# conda activate idx_creator
# python run_dt_place.py --wl 39 --rtg 1.1 --ecfg 27 --sidx 202
# conda activate idx_creator
# python run_dt_place.py --wl 39 --rtg 1.1 --ecfg 36 --sidx 203


# conda activate idx_creator
# python run_dt_place.py --wl 40 --rtg 1.1 --ecfg 100 --sidx 200
# conda activate idx_creator
# python run_dt_place.py --wl 40 --rtg 1.1 --ecfg 101 --sidx 201
# conda activate idx_creator
# python run_dt_place.py --wl 40 --rtg 1.1 --ecfg 27 --sidx 202
# conda activate idx_creator
# python run_dt_place.py --wl 40 --rtg 1.1 --ecfg 36 --sidx 203


python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 11 --ecfg 1 --sidx 230 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 11 --ecfg 3 --sidx 231 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 11 --ecfg 100 --sidx 232 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 11 --ecfg 101 --sidx 233 --is_eval_only --rtg 2



python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 13 --ecfg 1 --sidx 230 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 13 --ecfg 3 --sidx 231 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 13 --ecfg 100 --sidx 232 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-05-08-33-0.941.pkl" --wl 13 --ecfg 101 --sidx 233 --is_eval_only --rtg 2

========
python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 11 --ecfg 1 --sidx 234 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 11 --ecfg 3 --sidx 235 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 11 --ecfg 100 --sidx 236 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 11 --ecfg 101 --sidx 237 --is_eval_only --rtg 2



python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 35 --ecfg 1 --sidx 234 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 35 --ecfg 3 --sidx 235 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 35 --ecfg 100 --sidx 236 --is_eval_only --rtg 2

python run_dt_place.py --p "amd_epyc7543_2s_8n" --mpath "save_models/amd_epyc7543_2s_8n/0/2024-10-25-08-13-41-0.966.pkl" --wl 35 --ecfg 101 --sidx 237 --is_eval_only --rtg 2





