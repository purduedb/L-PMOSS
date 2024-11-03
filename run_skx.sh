# For training 
python run_dt_place.py --p "intel_skx_4s_8n" --mpath "None" --wl 12 --ecfg 100 --sidx 259 --rtg 2

# For inference 
python run_dt_place.py --p "intel_skx_4s_8n" --mpath "save_models/intel_skx_4s_8n/0/2024-10-25-18-14-13-0.992.pkl" --wl 12 --ecfg 100 --sidx 259 --is_eval_only --rtg 2