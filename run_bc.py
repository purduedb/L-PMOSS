import numpy as np
import d3rlpy
import os
import glob
import  pandas as pd
import os
import logging
import argparse
import time
import math
from mingpt.utils import set_seed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from mingpt.model_placement import GPT, GPTConfig
from mingpt.trainer_placement import Trainer, TrainerConfig
from yr_utils import gen_token, gen_token_for_eval, gen_token_for_all, gen_token_for_eval_for_all, collect_stats_about_offline_dataset
from torch.utils.data.dataloader import DataLoader
from pmoss_configs import *
from d3rlpy.metrics import *
import torch
import numpy as np
from yr_utils import *
import torch.nn as nn
from d3rlpy.models.encoders import Encoder
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.logging import LoggerAdapterFactory
from d3rlpy.optimizers.optimizers import AdamFactory
from d3rlpy.optimizers.lr_schedulers import CosineAnnealingLRFactory
from d3rlpy.constants import LoggingStrategy

class SilentAdapter(LoggerAdapterFactory):
    def create(self, *args, **kwargs):
        class _SilentLogger:
            def __getattr__(self, name):
                # any call like logger.info(...), logger.add_scalar(...), etc.
                return lambda *a, **kw: None
        return _SilentLogger()
    
def get_parameter_number(bc_model):
    """Count all trainable parameters in BC model"""
    if bc_model.impl is None:
        print("BC model not built yet")
        return

    total_params = 0

    # Count imitator (policy) parameters
    if hasattr(bc_model.impl, '_modules') and hasattr(bc_model.impl._modules, 'imitator'):
        imitator = bc_model.impl._modules.imitator

        # Encoder parameters
        if hasattr(imitator, '_encoder'):
            encoder_params = sum(p.numel() for p in imitator._encoder.parameters() if p.requires_grad)
            print(f"  Encoder: {encoder_params:,} parameters")
            total_params += encoder_params

        # Policy head parameters (after encoder)
        if hasattr(imitator, '_fc'):
            head_params = sum(p.numel() for p in imitator._fc.parameters() if p.requires_grad)
            print(f"  Policy Head: {head_params:,} parameters")
            total_params += head_params

        # If we can't break it down, just count all imitator params
        if total_params == 0:
            total_params = sum(p.numel() for p in imitator.parameters() if p.requires_grad)
            print(f"  Imitator (total): {total_params:,} parameters")

    print(f"BC Model Total Trainable Parameters: {total_params:,}")
    return total_params


print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(torch.__version__)       # Check PyTorch version
print(torch.version.cuda)      # Check CUDA version

try:
    x = torch.tensor([1.0]).to("cuda")
    print("CUDA is working!")
except Exception as e:
    print("CUDA is not available:", e)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=100)  # my=> 100 in stead of 256
parser.add_argument('--epochs', type=int, default=100)
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
parser.add_argument('--mpath', type=str, default="/scratch/gilbreth/xxxxxxx/save_models/_d3rlpy_bc_models/2025-10-07-23-01-12-0.553.pt")
parser.add_argument('--dbidx', type=int, default=0)
parser.add_argument('--idxkb', type=str, default="kb_b")  # kb_b__ was for amd with the fsanitizer stuff
parser.add_argument('--ablation_study', action='store_true', help='Enable ablation study mode')
parser.add_argument('--ablation_param', type=str, choices=['num_layer', 'num_head', 'num_embedding'], 
                   help='Which parameter to ablate (num_layer, num_head, num_embedding)')
parser.add_argument('--generalization_study', action='store_true', help='Enable generalization study mode')
parser.add_argument('--exclude_machine', type=str, help='Machine to exclude from training for generalization study')
parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')

parser.add_argument('--n_embd', type=int, default=512, help='Embedding dimension')
parser.add_argument('--model_type', type=str, default='reward_conditioned', choices=['reward_conditioned', 'naive'], help='Type of model to use (reward_conditioned or naive)')
parser.add_argument('--finetuning', action='store_true', help='Enable finetuning mode (saves models to separate directory)')
parser.add_argument('--save_path', type=str, default='/scratch/gilbreth/xxxxxxx/save_models/_d3rlpy_bc_models/', help='Directory to save trained models')

# changed kb_b for idx kb and kbs to kbs_train
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


workload = args.wl
eval_start_cfg = args.ecfg
save_idx = args.sidx
rtg_scale = args.rtg
cfg_to_start_with = args.ecfg
db_index = args.dbidx
db_index_kb_folder = args.idxkb

cd=(8,12)
nf=15
nmf=24
glb_exp_config = []
for p in [
    # "intel_skx_4s_8n", 
    # "amd_epyc7543_2s_8n",
    # "amd_epyc7543_2s_2n", 
    # "intel_sb_4s_4n",
    # "nvidia_gh_1s_1n",
    # "ibm_power_2s_2n",
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
                        ablation_study = args.ablation_study,
                        ablation_param = args.ablation_param,
                        generalization_study = args.generalization_study,
                        exclude_machine = args.exclude_machine,
                        n_layer = args.n_layer,
                        n_head = args.n_head,
                        n_embd = args.n_embd,
                       )
    glb_exp_config.append(exp_config)

# collect_stats_about_offline_dataset(glb_exp_config)


obss, obss_s, obss_mask, actions, stepwise_returns, rtgs, done_idxs, timesteps, meta_data, lengths, benchmarks \
    = gen_token_for_all(glb_exp_config)


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

# Convert PMOSS data to d3rlpy format
print("============================================================================================================")
print("Converting to d3rlpy format...")

# Determine action space size (8×12 = 96)
action_space_size = glb_exp_config[0].chassis_dim[0] * glb_exp_config[0].chassis_dim[1]
print(f"Target action space size: {action_space_size}")

# Flatten the observation data for d3rlpy
# d3rlpy expects observations as 2D arrays (n_samples, n_features)
observations = []
d3rl_actions = []
d3rl_rewards = []
d3rl_terminals = []

# Process training data
n_samples = obss.shape[0]
print(f"Processing {n_samples} samples...")

# Flatten observations: combine obss, obss_s, and obss_mask
for i in range(n_samples):
    # Flatten each observation component
    obs_flat = obss[i].flatten()  # Shape: (grid*grid,)
    obs_s_flat = obss_s[i].flatten()  # Shape: (grid*grid*features,)
    obs_mask_flat = obss_mask[i].flatten()  # Shape: (grid*grid,)
    meta_data_flat = meta_data[i].flatten()  # Shape: (num_meta_features,)
    
    # Concatenate all observation components
    full_obs = np.concatenate([obs_flat, obs_s_flat, obs_mask_flat, meta_data_flat])
    observations.append(full_obs)
    
    # Actions are already in the right format
    d3rl_actions.append(actions[i])
    
    # Use stepwise returns as rewards
    d3rl_rewards.append(stepwise_returns[i])
    
    # Create terminal flags (end of episode)
    # In PMOSS, each episode has 256 positioning steps, so terminal occurs every 256 steps
    is_terminal = ((i + 1) % 256 == 0)
    d3rl_terminals.append(is_terminal)

# Convert to numpy arrays
observations = np.array(observations, dtype=np.float32)
d3rl_actions = np.array(d3rl_actions, dtype=np.int64)
d3rl_rewards = np.array(d3rl_rewards, dtype=np.float32).flatten()
d3rl_terminals = np.array(d3rl_terminals, dtype=bool)

# Force action space to be 96 by adding dummy transitions for unused actions
unique_actions = np.unique(d3rl_actions)
max_action = action_space_size - 1  # 95 (for 96 total actions: 0-95)
print(f"Original unique actions: {len(unique_actions)} (range: {unique_actions.min()}-{unique_actions.max()})")

# Add minimal dummy transitions for missing actions to force action space = 96
missing_actions = []
for action_id in range(action_space_size):
    if action_id not in unique_actions:
        missing_actions.append(action_id)

if missing_actions:
    print(f"Adding {len(missing_actions)} dummy transitions for missing actions: {missing_actions[:10]}...")
    
    # Create dummy observations (copy the first observation)
    dummy_obs = np.tile(observations[0], (len(missing_actions), 1))
    dummy_rewards = np.zeros(len(missing_actions), dtype=np.float32)
    dummy_terminals = np.ones(len(missing_actions), dtype=bool)  # Mark as terminal
    dummy_actions = np.array(missing_actions, dtype=np.int64)
    
    # Append dummy data
    observations = np.vstack([observations, dummy_obs])
    d3rl_actions = np.concatenate([d3rl_actions, dummy_actions])
    d3rl_rewards = np.concatenate([d3rl_rewards, dummy_rewards])
    d3rl_terminals = np.concatenate([d3rl_terminals, dummy_terminals])

print(f"Final data shapes (with action space = {action_space_size}):")
print(f"Observations: {observations.shape}")
print(f"Actions: {d3rl_actions.shape} (unique: {len(np.unique(d3rl_actions))})")
print(f"Rewards: {d3rl_rewards.shape}")
print(f"Terminals: {d3rl_terminals.shape}")
print(f"Action range: {d3rl_actions.min()}-{d3rl_actions.max()}")

# Create d3rlpy dataset
dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=d3rl_actions,
    rewards=d3rl_rewards,
    terminals=d3rl_terminals,
)

print(f"Created d3rlpy dataset with {len(observations)} transitions")
print(f"Dataset episodes: {len(dataset.episodes)}")
print(f"Total transitions across all episodes: {sum(len(episode) for episode in dataset.episodes)}")
print("============================================================================================================")

# Completely disable all logging except CRITICAL errors
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.CRITICAL,  # Changed from INFO to CRITICAL
)

# Disable d3rlpy logging completely
logging.getLogger('d3rlpy').disabled = True
logging.getLogger('d3rlpy.algos').disabled = True
logging.getLogger('d3rlpy.metrics').disabled = True
logging.getLogger('d3rlpy.dataset').disabled = True
logging.getLogger('d3rlpy.models').disabled = True
logging.getLogger('d3rlpy.preprocessing').disabled = True
logging.getLogger('d3rlpy.gpu').disabled = True


# D3RLPY Behavior Cloning Setup
print("============================================================================================================")
print("Setting up d3rlpy Behavior Cloning...")

# Action space size already determined above
print(f"Action space size: {action_space_size}")

# Setup d3rlpy Discrete BC algorithm
class PMOSSStateEncoder(nn.Module):
    def __init__(self, input_shape, n_embd, num_features, num_mfeatures=nmf):
        super().__init__()
        chassis_dimx = cd[0]
        chassis_dimy = cd[1]

        self.c, self.h, self.w = 3+num_features, chassis_dimx, chassis_dimy  # e.g. (3 + num_features, 64, 64)
        self.num_features = num_features
        self.num_mfeatures = num_mfeatures

        self.state_encoder_s = nn.Sequential(
            nn.Conv2d(self.c, 16, 8, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
    
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, n_embd)
        )

        self.meta_encoder_s = nn.Sequential(
            nn.Linear(self.num_mfeatures+2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_mfeatures)
            )

    def forward(self, x):
        x1_ = x[:, :self.c*self.h*self.w]
        x2_ = x[:, self.c*self.h*self.w:]

        x1_ = x1_.view(x.size(0), self.c, self.h, self.w)
        state_embeddings = self.state_encoder_s(x1_)

        if not(self.num_mfeatures == 0):
            meta_embeddings = self.meta_encoder_s(
                x2_.reshape(-1, self.num_mfeatures+2)
                )
            # print(meta_embeddings.shape, state_embeddings.shape)
            state_embeddings = torch.cat((state_embeddings, meta_embeddings[:, :].reshape(-1, self.num_mfeatures)), dim = 1)

        state_embeddings = nn.Tanh()(state_embeddings)

        return state_embeddings
    






class PMOSSStateEncoderFactory(EncoderFactory):
    TYPE = "pmoss_state"

    def __init__(self, n_embd=128, num_features=nf, num_meta_features=nmf, use_large=False, use_xl=False):
        self.n_embd = n_embd
        self.num_features = num_features
        self.num_meta_features = num_meta_features
        self.use_large = use_large
        self.use_xl = use_xl

    def create(self, observation_shape):
        return PMOSSStateEncoder(observation_shape, self.n_embd, self.num_features,
                                     self.num_meta_features)
    
    def get_type(self):
        return self.TYPE


encoder_factory = PMOSSStateEncoderFactory(
    n_embd=args.n_embd,
    num_features=nf,
    num_meta_features=nmf,
)

# Calculate training steps first (needed for LR scheduler)
samples_per_epoch = len(observations)
n_steps_per_epoch = samples_per_epoch // args.batch_size
n_epochs = args.epochs
n_steps = n_epochs * n_steps_per_epoch
save_interval = n_steps_per_epoch * 1

# Use different save paths for finetuning vs training from scratch
if args.finetuning:
    model_save_path = os.path.join(args.save_path, glb_exp_config[0].processor)
    print("Finetuning mode enabled - models will be saved to:", model_save_path)
else:
    model_save_path = args.save_path
    print("Training mode - models will be saved to:", model_save_path)

os.makedirs(model_save_path, exist_ok=True)
print(n_steps, n_steps_per_epoch, save_interval)

# Determine learning rate based on whether we're loading a checkpoint
initial_lr = 6e-4 if model_path is not None and os.path.exists(model_path) else 6e-4

bc_config = d3rlpy.algos.DiscreteBCConfig(
    learning_rate=initial_lr,
    batch_size=args.batch_size,
    encoder_factory=encoder_factory,
    optim_factory=AdamFactory(
        weight_decay=0.0,
        lr_scheduler_factory=CosineAnnealingLRFactory(T_max=n_steps, eta_min=6e-5)  # Cosine decay
    ),
    beta=0.5  # Label smoothing: helps prevent overconfidence (default is 1.0)
)

# Print training mode information
if model_path and os.path.exists(model_path):
    mode_str = "fine-tuning" if args.finetuning else "resuming training"
    print(f"Using initial learning rate: {initial_lr} ({mode_str} from checkpoint)")
else:
    print(f"Using initial learning rate: {initial_lr} (training from scratch)")
bc = bc_config.create(
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

# Load checkpoint if model_path is provided
if model_path is not None and os.path.exists(model_path):
    print(f"Loading BC checkpoint from: {model_path}")
    bc.build_with_dataset(dataset)
    bc.load_model(model_path)
    print("BC checkpoint loaded successfully! Resuming training...")
else:
    bc.build_with_dataset(dataset)

discrete_action_match_evaluator = d3rlpy.metrics.DiscreteActionMatchEvaluator()
print(type(bc.impl))
print(bc)


# Print parameter count (model already built above)
print("\n========== BC Model Parameters ==========")
get_parameter_number(bc)
print("=========================================\n")
# exit(0)

# ========== TRAINING MODE ========== #
if not(args.is_eval_only):
    print("Starting d3rlpy Behavior Cloning training with accuracy-based saving...")
    def save_checkpoint_callback(bc_model, step, epoch, evaluator, save_dir, min_accuracy):
        # Get current learning rate from optimizer
        current_lr = None
        if hasattr(bc_model.impl, '_optim') and bc_model.impl._optim is not None:
            current_lr = bc_model.impl._optim.param_groups[0]['lr']
        accuracy = evaluator(bc_model, dataset)
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # Print with learning rate
        if current_lr is not None:
            print(f"Epoch {epoch} | LR: {current_lr:.6f} | Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        else:
            print(f"Epoch {epoch} | Accuracy: {accuracy:.4f} ({accuracy:.1%})")

        # Build custom filename
        if accuracy >= min_accuracy:
            filename = f"{timestamp}-{accuracy:.3f}.d3"
            path = os.path.join(save_dir, filename)
            bc_model.save_model(path)
            print(f"Saved checkpoint: {path}")
        else:
            filename = f"{timestamp}-{accuracy:.3f}_lowaccuracy.d3"
    
    logging.getLogger("d3rlpy").setLevel(logging.ERROR)  
    logging.getLogger("d3rlpy").propagate = False
    bc.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        evaluators={'action_match': discrete_action_match_evaluator},
        experiment_name=None,
        with_timestamp=False,
        epoch_callback=lambda model, step, epoch: save_checkpoint_callback(
            bc_model=model,
            step=step,
            epoch=epoch,
            evaluator=discrete_action_match_evaluator,
            save_dir=model_save_path,
            min_accuracy=0.1
        ),
        save_interval=10000000,
        logging_steps=1e15,
        logging_strategy=LoggingStrategy.EPOCH,
        logger_adapter=SilentAdapter(),
        show_progress=False
    )
    
    print("Training completed. Evaluating final model...")
    
    final_score = discrete_action_match_evaluator(bc, dataset)
    print(f"Final action match accuracy: {final_score:.4f} ({final_score:.1%})")
    


obss_, obss_s_, obss_mask_, actions_, stepwise_returns_, rtgs_, done_idxs_, timesteps_, meta_data_, lengths_, benchmarks_ \
    = gen_token_for_eval_for_all(glb_exp_config)

context_length = glb_exp_config[0].cnt_grid_cells
test_dataset = StateActionReturnDataset(
    glb_exp_config[0],
    obss_, context_length*3, actions_,
    done_idxs_, rtgs_, timesteps_, meta_data_, obss_s_,
    obss_mask_, benchmarks_, stepwise_returns_, lengths_
)
print("Using loaded BC model for evaluation (inference mode)..." if args.is_eval_only else "Using best/final BC model for evaluation (training completed)...")
print(f"Model path: {model_path if args.is_eval_only else 'N/A'}")

# =====================
# DT-style BC policy evaluation (true empty-state rollout)
# =====================
def evaluate_bc_policy_rollout_dt_style(bc_model, exp_config, test_dataset):
    """
    DT-style sequential rollout for BC model: start from empty state, sequentially predict actions, update state with env_update,
    and compute action match accuracy. Closely mimics Decision Transformer evaluation.
    Returns accuracy and predicted action sequences.
    """
    loader = DataLoader(test_dataset, shuffle=True, pin_memory=True,
                batch_size=args.batch_size,
                num_workers=2
                )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # The recent observation: the very last observation
    pbar = enumerate(loader)
    for it, (x, y, r, t, m_x, b, st, cir, l) in pbar:   
        x = x.to(device)[0, -1, :]  # states: my=> (batch, context, 8*grid*grid)
        m_x = m_x.to(device)[0, -1, :]  # meta: my=> (batch, context, 6)
        y = y.to(device)[0, -1, :]  # action: my=> (batch, context, 1)
        r = r.to(device)[0, :, :, :]  # rtg: my=> (batch, context, 1, 1) should be (batch, context, 1)
        t = t.to(device)[0, -1, :]  # ts: my=> (batch, context, 1)
        b = b.to(device)[0, -1, :, :]  # benchmark: my=> (batch, context, 1, 1)
        st = st.to(device)[0, :, :, :]  # stepwise returns: my=> (batch, context, 1, 1)
        cir = cir.to(device)[0]  # circuit: my=> (batch, 768)
        l = l.to(device)[0]  # where to stop in a batch: my=> (batch, context)

    print(x.shape, m_x.shape, y.shape, r.shape, t.shape, b.shape, st.shape, cir.shape, l.shape)

    # Get dimensions
    chassis_dimx = exp_config.chassis_dim[0]
    chassis_dimy = exp_config.chassis_dim[1]
    num_features = exp_config.num_features + 1  # +1 for grid index
    seq_len = exp_config.cnt_grid_cells
    
    # Use only the first sample's ground-truth actions for comparison
    # --- Initial empty state setup (mimic DT get_returns logic) ---
    state_obs = torch.tensor(np.full((1, chassis_dimx, chassis_dimy), False))
    state_obs_s = torch.tensor(np.full((num_features, chassis_dimx, chassis_dimy), 0, dtype=np.float64))
    # state_obs = torch.zeros((1, chassis_dimx, chassis_dimy), dtype=torch.bool, device=device)
    # state_obs_s = torch.zeros((num_features, chassis_dimx, chassis_dimy), dtype=torch.float32, device=device)

    cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
    state_obs_mask = np.full((chassis_dimx * chassis_dimy,), False)
    obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
    chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
    bound_core = int(exp_config.cnt_grid_cells / exp_config.machine.num_worker)+1
    obs_mask_core[np.array(chassis_act_).astype(int)] = bound_core
		
    """If you want to restrict the allocation to only worker cores"""
    mask_already_full = np.where(obs_mask_core==0)
    state_obs_mask[mask_already_full] = True
    # """If you want to balance the load across cpus:Do not pass the binary masks only"""
    # state_obs_mask = obs_mask_core

    state_obs_mask = np.reshape(state_obs_mask, (1, chassis_dimx, chassis_dimy))
    state_obs_mask = torch.tensor(state_obs_mask)
		
    # print(state_obs_mask)
    state = torch.cat((state_obs, state_obs_s, state_obs_mask), 0).view(-1, chassis_dimx, chassis_dimy)

    reward_sum = 0
    done = False 
		
    # meta_state = torch.zeros_like(m_x)
    meta_state = m_x
		
    # print(state.shape)
    # print(reward_sum)
    # print(done)
    # print(meta_state.shape)
    
    
    score_sum = 0
    assert reward_sum == 0
		
    rewards = []
    probs = []
    
    state = state.type(torch.float32).to(device).unsqueeze(0)
    meta_state = meta_state.type(torch.float32).to(device).unsqueeze(0)

    # Set model to evaluation mode (disables dropout, batchnorm updates)
    # For d3rlpy models, need to access the underlying PyTorch modules
    bc_model.impl._modules.imitator.eval()  # CRITICAL: Must set to eval mode for BatchNorm with batch_size=1

    # state_obs_mask = torch.zeros((1, chassis_dimx, chassis_dimy), dtype=torch.bool, device=device)
    # state = torch.cat((state_obs, state_obs_s, state_obs_mask), 0).view(-1, chassis_dimx, chassis_dimy)
    # Initial meta_state: zeros
    # meta_state = torch.zeros((1, exp_config.num_meta_features), dtype=torch.float32, device=device)
    # Initial RTG: 0 (not used for BC)
    # current_rtg = torch.zeros(1, dtype=torch.float32)
    # obs_mask_core = torch.ones(chassis_dimx * chassis_dimy, dtype=torch.int32)
    pred_actions = []
    done = False
    # Log time of this prediction
    start_time = time.time()
    # --- Rollout loop ---
    rtgs = [0.0]
    current_rtg = torch.tensor(rtgs)
    for t in range(seq_len):
        # Keep tensors on GPU - no CPU/GPU transfer in loop (much faster!)
        state_ = state[-1].view(1, -1)
        meta_state_ = meta_state[-1].view(1, -1)

        # Concatenate on GPU
        obs_tensor = torch.cat([state_, meta_state_], dim=1)
        
        # action = bc_model.predict(obs)[0]
        with torch.no_grad():
            logits = bc_model.impl._modules.imitator(obs_tensor).logits
            mask_tensor = torch.from_numpy(np.array(obs_mask_core) == 0).float().to(logits.device)
            logits = logits - 1.0e8 * mask_tensor
            logits_actions = logits.argmax(dim=1)
            
        if isinstance(logits_actions, int):
            pred_actions += [logits_actions]
        else:
            pred_actions += [logits_actions.item()]
        
        state, current_rtg, done, meta_state, obs_mask_core = env_update(
            x, m_x, st,  
            pred_actions, state, meta_state, current_rtg, exp_config,
            exp_config.eval_start_cfg, obs_mask_core)
        

        if done:
            break
    pred_actions = np.array(pred_actions)
    """actions here are hw positions of the workers"""
    retrieve_config(exp_config, pred_actions, args.sidx)
    print("REFINE ACTION")
    print("=====================ALL DONE!=====================")
    print("Total rollout time:", time.time() - start_time)
    return 

bc_actions = evaluate_bc_policy_rollout_dt_style(bc, glb_exp_config[0], test_dataset)





