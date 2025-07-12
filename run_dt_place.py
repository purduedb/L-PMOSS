import os
import logging
import argparse
from mingpt.utils import set_seed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from mingpt.model_placement import GPT, GPTConfig
from mingpt.trainer_placement import Trainer, TrainerConfig
from yr_utils import gen_token, gen_token_for_eval, gen_token_for_all, gen_token_for_eval_for_all
from torch.utils.data.dataloader import DataLoader
from pmoss_configs import *

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(torch.__version__)       # Check PyTorch version
print(torch.version.cuda)      # Check CUDA version

try:
    x = torch.tensor([1.0]).to("cuda")
    print("CUDA is working!")
except Exception as e:
    print("CUDA is not available:", e)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=100)  # my=> 100 in stead of 256
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--is_eval_only', action='store_true')
parser.add_argument('--no_eval_only', action='store_false')
parser.add_argument('--test_all_macro', action='store_true')
parser.add_argument('--start_cfg', type=int, default=30)
parser.add_argument('--rtg', type=float, default=1.1)

parser.add_argument('--wl', type=int, default=11)
parser.add_argument('--ecfg', type=int, default=30)
parser.add_argument('--sidx', type=int, default=1)
parser.add_argument('--p', type=str, default="amd_epyc7543_2s_8n")
parser.add_argument('--mpath', type=str, default="/scratch/gilbreth/yrayhan/save_models/amd_epyc7543_2s_8n/0/2024-10-23-07-27-55-0.949.pkl")
parser.add_argument('--dbidx', type=int, default=0)
parser.add_argument('--idxkb', type=str, default="kb_b")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

set_seed(args.seed)
# seq_len = args.context_length       # total number of grids
# rtg_scale = args.rtg                # e.g., 1.1, 1.2, ...
# cfg_to_start_with = args.start_cfg  # necessary for inference (recent snaps)

model_path = None if args.mpath == "None" else args.mpath

class StateActionReturnDataset(Dataset):

    def __init__(self, exp_config, data, block_size, actions, done_idxs, rtgs, 
            timesteps, meta_data = None, obss_wire = None, obss_mask = None, benchmarks = None,
            stepwise_returns = None, lengths = None):
        
        assert block_size % 3 == 0
    
        self.block_size = block_size
        self.seq_len = self.block_size // 3
        self.vocab_size = exp_config.chassis_dim[0]*exp_config.chassis_dim[1]
        
        self.data = data
        self.actions = actions
        print("data raw shape", data.shape)
        self.done_idxs = done_idxs
        self.meta_data = meta_data
        # print("meta_data raw shape", meta_data.shape)
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.obss_wire = obss_wire
        self.obss_mask = obss_mask
        self.benchmarks = benchmarks
        self.stepwise_returns = stepwise_returns
        self.lengths = lengths
    
    def __len__(self):
        return len(self.data)//self.seq_len

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        idx = idx * self.seq_len
        done_idx = idx + self.seq_len
        if self.obss_mask is None:
            states = torch.tensor(np.array(self.data[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        else:
            tmp_obss = torch.tensor(np.array(self.data[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1)
            tmp_obss_wire = torch.tensor(np.array(self.obss_wire[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1)
            tmp_obss_mask = torch.tensor(np.array(self.obss_mask[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1)
            
            states = torch.cat((tmp_obss, tmp_obss_wire, tmp_obss_mask), dim=1)
            # => h/w my change for hw
            # states = torch.cat((tmp_obss, tmp_obss_mask), dim=1)

        meta_states = torch.tensor(np.array(self.meta_data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        benchmarks = torch.tensor(self.benchmarks[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        stepwise_returns = torch.tensor(self.stepwise_returns[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        benchmark_id = int(self.benchmarks[idx][0])
        # circuit_feas_for_benchmark = torch.tensor(circuit_feas[benchmark_id], dtype = torch.float32) 
        circuit_feas_for_benchmark = torch.randn(768)
        
        length = torch.zeros((block_size,), dtype=torch.bool)
        length[:int(self.lengths[idx][0])] = 1
        return states, actions, rtgs, timesteps, meta_states, \
            benchmarks, stepwise_returns, circuit_feas_for_benchmark, length

# p=args.p
# cd=(1,1)
# nf=-1
# nmf=0
# if p == "intel_skx_4s_8n":
#     cd = (8,12)
#     nf=15
#     nmf=24  # 16 + 8 
# elif p == "amd_epyc7543_2s_8n":
#     cd = (8,8)
#     nf=12
#     nmf=0
# elif p == "nvidia_gh_1s_1n":
#     cd = (8,9)
#     nf=12
#     nmf=0
# elif p == "amd_epyc7543_2s_2n":
#     cd = (8,8)
#     nf=12
#     nmf=0
#     # "Needs to be updated"
# elif p == "intel_sb_4s_4n":
#     cd = (8,8)
#     nf=15
#     nmf=16
# elif p == "all":
#     cd = (8,12)
#     nf=15
#     nmf=24

workload = args.wl
eval_start_cfg = args.ecfg
save_idx = args.sidx
rtg_scale = args.rtg
cfg_to_start_with = args.ecfg
db_index = args.dbidx
db_index_kb_folder = args.idxkb


# args.is_eval_only = True
# model_path = "/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl"

cd=(8,12)
nf=15
nmf=24
glb_exp_config = []
for p in [
    # "intel_skx_4s_8n", 
    # "ibm_power_2s_2n",
    # "amd_epyc7543_2s_8n",
    # "amd_epyc7543_2s_2n", 
    # "intel_sb_4s_4n",
    "nvidia_gh_1s_1n",
    # "intel_ice_2s_2n",
]:
    exp_config = ExpConfig(processor=p, 
                        chassis_dim=cd, 
                        index=db_index,
                        workload=workload,
                        num_features=nf, 
                        num_meta_features=nmf, 
                        cnt_grid_cells=256, 
                        cfg_par=4, 
                        per_cfg_sample=7, # 5
                        policy_dim = (16, 16), 
                        rtg_scale=rtg_scale,
                        rtg_div=100000,
                        eval_start_cfg=eval_start_cfg,
                        idx_kb_folder=db_index_kb_folder,
                        save_idx = save_idx,
                       )
    glb_exp_config.append(exp_config)
obss, obss_s, obss_mask, actions, stepwise_returns, rtgs, done_idxs, timesteps, meta_data, lengths, benchmarks \
    = gen_token_for_all(glb_exp_config)

# obss, obss_s, obss_mask, actions, stepwise_returns, rtgs, done_idxs, timesteps, meta_data, lengths, benchmarks \
#     = gen_token(exp_config)


# They should have stuff of all 



# cut = int(obss.shape[0]*0.5)
# obss = obss[:cut]
# obss_s = obss_s[:cut]
# obss_mask = obss_mask[:cut]
# actions = actions[:cut]
# stepwise_returns = stepwise_returns[:cut]
# rtgs = rtgs[:cut]
# done_idxs = done_idxs[:cut]
# timesteps = timesteps[:cut]
# meta_data = meta_data[:cut]
# lengths = lengths[:cut]
# benchmarks = benchmarks[:cut]
# DF = pd.DataFrame(np.reshape(obss_s, (obss_s.shape[0], -1))[:1000]) 
# DF.to_csv("data1.csv")


# => my 
obss_, obss_s_, obss_mask_, actions_, stepwise_returns_, rtgs_, done_idxs_, timesteps_, meta_data_, lengths_, benchmarks_ \
    = gen_token_for_eval_for_all(glb_exp_config)
# obss_, obss_s_, obss_mask_, actions_, stepwise_returns_, rtgs_, done_idxs_, timesteps_, meta_data_, lengths_, benchmarks_ \
#     = gen_token_for_eval(exp_config)

print("============================================================================================================")
print("create dataset finish.")
print("obss shape = ", obss.shape)  # (records, 1, grid, grid) => False, true
print("obss_wire shape = ", obss_s.shape)  # (records, 1, grid, grid)  => float
print("obss_mask shape = ", obss_mask.shape)  # (records, 1, grid, grid)  => True, false

print("actions shape = ", actions.shape)  # (records, ) => int
# print("returns shape = ", returns.shape)  # (101, 1) => float
print("done_idxs shape = ", done_idxs.shape)  # (100, ) => 256 * i => 256, 512, 768
print("rtgs shape = ", rtgs.shape)  # (records, )  => float

print("timesteps shape = ", timesteps.shape)  # (records, )  => [0-255][0-255][0-255]
if not(exp_config.num_meta_features) == 0:
    print("meta_data shape = ", meta_data.shape)  # (records, 6)  => negative values

print("benchmarks shape = ", benchmarks.shape)  # (records, 1)  => all 0s`
print("stepwise_returns shape = ", stepwise_returns.shape)  # (records, 1)  => float
print("lengths shape = ", lengths.shape)  # (records, 1) => 63s and 0s
print("============================================================================================================")


print("create dataset finish.")


# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# my=>
context_length = exp_config.cnt_grid_cells
train_dataset = StateActionReturnDataset(
    exp_config,
    obss, context_length*3, actions, 
    done_idxs, rtgs, timesteps, meta_data, obss_s, 
    obss_mask, benchmarks, stepwise_returns, lengths
    )
test_dataset = StateActionReturnDataset(
    exp_config,
    obss_, context_length*3, actions_, 
    done_idxs_, rtgs_, timesteps_, meta_data_, obss_s_, 
    obss_mask_, benchmarks_, stepwise_returns_, lengths_
    )

# To check if loading is done correctly
# loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
#                                 batch_size=32)
# pbar = enumerate(loader)
# for it, (x, y, r, t, m_x, b, st, cir, l) in pbar:
#     # states, actions, rtgs, timesteps, meta_states, benchmarks, stepwise_returns, circuit_feas_for_benchmark, length
    
#     # place data on the correct device
#     x = x  # my=> (batch, context, 8*grid*grid)
#     m_x = m_x  # my=> (batch, context, 6)
#     y = y  # my=> (batch, context, 1)
#     r = r  # my=> (batch, context, 1, 1) should be (batch, context, 1)
#     t = t  # my=> (batch, context, 1)
#     b = b  # my=> (batch, context, 1, 1)
#     st = st  # my=> (batch, context, 1, 1)
#     cir = cir  # my=> (batch, 768)
#     l = l # my=> (batch, context)
#     print(x.shape, y.shape, r.shape, t.shape, m_x.shape, b.shape, st.shape, cir.shape, l.shape)
#     print(x[0, 5, :].view(-1, ))
#     print(r[0].view(-1, ))
#     zz = input()

# print("!!!! max(timesteps)", max(timesteps))


# Model tuning 
mconf = GPTConfig(
    train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8, n_embd=128, 
    model_type="reward_conditioned", max_timestep=max(timesteps))

model = GPT(mconf, exp_config)
# model_path = None
# model_path = "save_models/" + exp_config.processor + "/" + str(exp_config.index) + "/" + "2025-07-09-23-44-40-0.556.pkl"
# model_path = "/scratch/gilbreth/yrayhan/save_models/intel_sb_4s_4n/0/2025-07-10-17-29-23-0.952.pkl"
# print(model_path)

if model_path is not None:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    for k,v in state_dict.items():
        if "module." in k:
            state_dict[k.split('.', 1)[1]] = v
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict, strict = True)
model.eval()
get_parameter_number(model)

# initialize a trainer instance and kick off training
epochs = args.epochs



tconf = TrainerConfig(
    max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
    lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
    num_workers=1, seed=args.seed, model_type="reward_conditioned", max_timestep=max(timesteps),
    draw_placement = True, is_eval_only = args.is_eval_only,
    test_all_macro = args.test_all_macro)
print("trainerconfig finish")

# => my test_dataset in place of None
trainer = Trainer(model, train_dataset, test_dataset, tconf, cfg_to_start_with, exp_config)
print("trainer build finish")
trainer.train()
