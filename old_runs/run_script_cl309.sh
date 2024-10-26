#!/bin/bash

# For training
python run_dt_place.py --p "intel_skx_4s_8n" --mpath "save_models/intel_skx_4s_8n/0/2024-10-23-06-22-42-0.922.pkl"
# python run_dt_place.py --p "intel_skx_4s_8n" --mpath "save_models/intel_skx_4s_8n/0/2024-10-24-23-56-03-0.910.pkl" --wl 11 --ecfg 1 --sidx 204 --is_eval_only
# python run_dt_place.py --wl 32 --ecfg 100 --sidx 200
# python run_dt_place.py --wl 41 --ecfg 100 --sidx 200
# python run_dt_place.py --wl 32 --ecfg 19 --sidx 202
# python run_dt_place.py --wl 41 --ecfg 19 --sidx 202

# python run_dt_place.py --wl 32 --ecfg 1 --sidx 203
# python run_dt_place.py --wl 41 --ecfg 1 --sidx 203
# python run_dt_place.py --wl 32 --ecfg 3 --sidx 204
# python run_dt_place.py --wl 41 --ecfg 3 --sidx 204
# python run_dt_place.py --wl 32 --ecfg 31 --sidx 205
# python run_dt_place.py --wl 41 --ecfg 31 --sidx 205
# python run_dt_place.py --wl 32 --ecfg 41 --sidx 206
# python run_dt_place.py --wl 41 --ecfg 41 --sidx 206