# For training 
# python run_dt_place.py --p "intel_skx_4s_8n" --mpath "None" --wl 12 --ecfg 100 --sidx 259 --rtg 2

# For inference 
python run_dt_place.py --mpath "/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl" --wl 11 --ecfg 100 --sidx 200 --is_eval_only --rtg 2