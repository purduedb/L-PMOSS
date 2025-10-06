#!/bin/bash

module load conda
conda activate pmoss

# These are all for intel_skx_4s_8n hardware

base_model_nl_2_wo_ibm=/scratch/gilbreth/yrayhan/save_models/nl_2/0/2025-10-05-02-52-35-0.676.pkl
base_model_nl_4_wo_ibm=/scratch/gilbreth/yrayhan/save_models/nl_4/0/2025-10-05-04-24-55-0.858.pkl
base_model_nl_8_wo_ibm=/scratch/gilbreth/yrayhan/save_models/nl_8/0/2025-10-05-02-32-27-0.921.pkl

# Declare num_layer array 
num_layer=(2 4 8)
sidx_list=(1000 1001 1002)
for i in "${!num_layer[@]}"; do
  nl=${num_layer[$i]}
  sidx=${sidx_list[$i]}
  
  # Check if nl is 2 or 4 or 8 and set the corresponding model path
  if [ "$nl" -eq 2 ]; then
    base_model_wo_ibm=$base_model_nl_2_wo_ibm
  elif [ "$nl" -eq 4 ]; then
    base_model_wo_ibm=$base_model_nl_4_wo_ibm
  elif [ "$nl" -eq 8 ]; then
    base_model_wo_ibm=$base_model_nl_8_wo_ibm
  else
    echo "Invalid num_layer value: $nl"
    exit 1
  fi

  python run_dt_place.py \
    --mpath "$base_model_wo_ibm" \
    --wl 11 \
    --ecfg 100 \
    --sidx "$sidx" \
    --is_eval_only \
    --rtg 2 \
    --ablation_study \
    --ablation_param num_layer \
    --n_layer "$nl" 
done



# base_model_nh_2_wo_ibm=/scratch/gilbreth/yrayhan/save_models/nh_2/0/2025-10-05-08-16-10-0.927.pkl
# base_model_nh_4_wo_ibm=/scratch/gilbreth/yrayhan/save_models/nh_4/0/2025-10-05-09-06-36-0.918.pkl
# base_model_nh_16_wo_ibm=/scratch/gilbreth/yrayhan/save_models/nh_16/0/2025-10-05-14-26-12-0.918.pkl
# # Declare num_head array and follow the same procedure as above
# num_head=(2 4 16)
# sidx_list=(1003 1004 1005)
# for i in "${!num_head[@]}"; do
#   nh=${num_head[$i]}
#   sidx=${sidx_list[$i]}

#   # Check if nh is 2 or 4 or 16 and set the corresponding model path
#   if [ "$nh" -eq 2 ]; then
#     base_model_wo_ibm=$base_model_nh_2_wo_ibm
#   elif [ "$nh" -eq 4 ]; then
#     base_model_wo_ibm=$base_model_nh_4_wo_ibm
#   elif [ "$nh" -eq 16 ]; then
#     base_model_wo_ibm=$base_model_nh_16_wo_ibm
#   else
#     echo "Invalid num_head value: $nh"
#     exit 1
#   fi

#   python run_dt_place.py \
#     --mpath "$base_model_wo_ibm" \
#     --wl 11 \
#     --ecfg 100 \
#     --sidx "$sidx" \
#     --is_eval_only \
#     --rtg 2 \
#     --ablation_study \
#     --ablation_param num_head \
#     --n_head "$nh"
# done



# base_model_ne_32_wo_ibm=/scratch/gilbreth/yrayhan/save_models/ne_32/0/2025-10-05-11-42-25-0.231.pkl
# base_model_ne_64_wo_ibm=/scratch/gilbreth/yrayhan/save_models/ne_64/0/2025-10-05-14-50-40-0.580.pkl
# base_model_ne_256_wo_ibm=/scratch/gilbreth/yrayhan/save_models/ne_256/0/2025-10-05-14-10-15-0.927.pkl

# # Decare num_embedding array and follow the same procedure as above
# num_embedding=(32 64 256)
# sidx_list=(1006 1007 1008)
# for i in "${!num_embedding[@]}"; do
#   ne=${num_embedding[$i]}
#   sidx=${sidx_list[$i]}

#   # Check if ne is 32 or 64 or 256 and set the corresponding model path
#   if [ "$ne" -eq 32 ]; then
#     base_model_wo_ibm=$base_model_ne_32_wo_ibm
#   elif [ "$ne" -eq 64 ]; then
#     base_model_wo_ibm=$base_model_ne_64_wo_ibm
#   elif [ "$ne" -eq 256 ]; then
#     base_model_wo_ibm=$base_model_ne_256_wo_ibm
#   else
#     echo "Invalid num_embedding value: $ne"
#     exit 1
#   fi

#   python run_dt_place.py \
#     --mpath "$base_model_wo_ibm" \
#     --wl 11 \
#     --ecfg 100 \
#     --sidx "$sidx" \
#     --is_eval_only \
#     --rtg 2 \
#     --ablation_study \
#     --ablation_param num_embedding \
#     --n_embd "$ne"
# done
