#!/bin/bash

# For training 
# python run_dt_place.py --p "intel_skx_4s_8n" --mpath "None" --wl 12 --ecfg 100 --sidx 259 --rtg 2

# For inference 
# python run_dt_place.py --p "intel_skx_4s_8n" --mpath "save_models/intel_skx_4s_8n/0/2024-10-25-18-14-13-0.992.pkl" --wl 12 --ecfg 100 --sidx 259 --is_eval_only --rtg 2


module load conda
conda activate pmoss

python run_dt_place.py --mpath "/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl" --wl 12 --ecfg 100 --sidx 259 --rtg 2


# [11 12 44 45 16]
# for wk in 13;do
#   python run_dt_place.py --mpath "/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl" --wl $wk --ecfg 100 --sidx 205 --is_eval_only --rtg 2
# done
