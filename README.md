# P-MOSS: Scheduling Main-Memory Indexes Over NUMA Servers Using Next Token Prediction

This repo contains the code for the ML component of PMOSS. 

Running L-PMOSS
--------------------------------------------------------------------------------
### Requirements
```
pip install -r requirements.txt
```

Training 
--------------------------------------------------------------------------------
### Pre-training
Include all the machines that PMOSS should be trained on in run_dt_place.py: Line 151-158
```
python run_dt_place.py --mpath "None" 
```

### Post-training
Include only the target machine, e.g., intel_sb_4s_4n, that PMOSS should be trained on in run_dt_place.py: Line 151-158
```
python run_dt_place.py --mpath "/scratch/xxxxxx/xxxxxxxxxx/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl" 
```

### Inference
Include only the target machine, e.g., intel_sb_4s_4n, that PMOSS should infer for in run_dt_place.py: Line 151-158. 
```
python run_dt_place.py \
    --mpath "/scratch/xxxxxx/xxxxxxxxxx/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl" \
    --wl "$wk" \
    --ecfg 100 \
    --sidx "$sidx" \
    --is_eval_only \
    --rtg 2
```

### Implementation
```
L-PMOSS
├── README.md                  # Project README file
├── machines                   # The folder for the configs of the servers 
│   ├── hw_readme.cfg          # Describes the format of the cfg files of the machines in the 
                                 sub-folders
├── mingpt                     # Folder that includes the model architecture and the training 
                                 procedure 
│   ├── model_placement.py     # Logic of the model architecture
│   ├── trainer_placement.py   # Logic of the training and inference
│   ...
├── pmoss_machine_configs      # Root folder for the inferred scheduling policies 
│   ├── amd_epyc7543_2s_2n     # Inferred scheduling policy for AMD EPYC 7543 Server with NPS=1
│   │   ├── 11                 # Inferred scheduling policy for workload 11: YCSB-A
│   │   │   ├── c_200_256.txt  # Policy ID: 200, Number of index slices: 256
│   │   │   ...
│   │   ...
│   ...
├── machine_configs            # Root folder for the scheduling policies that PMOSS is trained on
│   ├── amd_epyc7543_2s_2n     # Scheduling policies for AMD EPYC 7543 Server with NPS=1
│   │   ├── c_200_256.txt      # Policy ID: 200, Number of index slices: 256
│   │   ...
│   ...
├── yr_utils.py                # Logic for Token generation, input processing to feed into GPT
├── run_dt_place.py            # The entry-point
├── pmoss_configs.py           # Logic for representing servers and different workload and sever 
                                 mappings
├── ...                
```



