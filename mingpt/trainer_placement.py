"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import time
import logging
import copy
from tqdm import tqdm
import numpy as np
import statistics
import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import infer_action
from yr_utils import *


logger = logging.getLogger(__name__)
seq_len = 256



strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
	def __init__(self, model, train_dataset, test_dataset, config, assuming_cfg_idx, exp_config):
		self.model = model
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.config = config
		self.x_0 = None
		self.m_x_0 = None
		self.training_step = 0
		self.best_acc = 0.0
		self.best_loss = 9999999999999
		self.assuming_cfg_idx = assuming_cfg_idx
		self.exp_config = exp_config
		strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
		
		self.device = 'cpu'
		if torch.cuda.is_available():
			self.device = torch.cuda.current_device()
			self.model = torch.nn.DataParallel(self.model).to(self.device)

	def save_checkpoint(self):
		# DataParallel wrappers keep raw model object in .module attribute
		raw_model = self.model.module if hasattr(self.model, "module") else self.model
		logger.info("saving %s", self.config.ckpt_path)

	def train(self):
		model, config = self.model, self.config
		raw_model = model.module if hasattr(self.model, "module") else model
		optimizer = raw_model.configure_optimizers(config)

		def run_epoch(split, epoch_num=0):
			is_train = split == 'train'
			
			model.train(is_train)
			data = self.train_dataset if is_train else self.test_dataset
						
			loader = DataLoader(
				data, shuffle=True, pin_memory=True, batch_size=config.batch_size,
				num_workers=config.num_workers)

			losses = []
			accs = np.zeros(0)
			pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
			
			if not is_train:
				model.eval()
			
			for it, (x, y, r, t, m_x, b, st, cir, l) in pbar:
				# states, actions, rtgs, timesteps, meta_states, benchmarks, stepwise_returns, circuit_feas_for_benchmark, length
				self.training_step += 1
				# place data on the correct device
				x = x.to(self.device)  # my=> (batch, context, 8*grid*grid)
				m_x = m_x.to(self.device)  # my=> (batch, context, 6)
				y = y.to(self.device)  # my=> (batch, context, 1)
				r = r.to(self.device)  # my=> (batch, context, 1, 1) should be (batch, context, 1)
				t = t.to(self.device)  # my=> (batch, context, 1)
				b = b.to(self.device)  # my=> (batch, context, 1, 1)
				st = st.to(self.device)  # my=> (batch, context, 1, 1)
				cir = cir.to(self.device)  # my=> (batch, 768)
				l = l.to(self.device)  # my=> (batch, context)
				x_0 = x[0]
				# print(x.shape, y.shape, r.shape, t.shape, m_x.shape, b.shape, st.shape, cir.shape, l.shape)

				
				# forward the model
				with torch.set_grad_enabled(is_train):
					logits, loss, acc = model(x, y, y, r, t, m_x, b, st, cir, l)
					loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
					# self.writer.add_scalar('loss', loss, self.training_step)
					acc = acc.mean()
					# self.writer.add_scalar('acc', acc, self.training_step)
					losses.append(loss.item())
					
				if is_train:
					# backprop and update the parameters
					model.zero_grad()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
					optimizer.step()
					# decay the learning rate based on our progress
					if config.lr_decay:
						self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
						if self.tokens < config.warmup_tokens:
							# linear warmup
							lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
						else:
							# cosine learning rate decay
							progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
							lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
						lr = config.learning_rate * lr_mult
						for param_group in optimizer.param_groups:
							param_group['lr'] = lr
					else:
						lr = config.learning_rate
					# self.writer.add_scalar('lr', lr, self.training_step)

					# report progress
					accs = np.append(accs, acc.cpu().numpy().mean())
					pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}.")
					
				# save model
				if accs.mean() > self.best_acc + 0.02 and accs.mean()>=0.2:
					self.best_acc = accs.mean()
					strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
					model.eval()
					raw_model = self.model.module if hasattr(self.model, "module") else self.model
					save_models_dir = "/scratch/gilbreth/yrayhan/save_models/" + self.exp_config.processor + "/" + str(self.exp_config.index)
					os.makedirs(save_models_dir, exist_ok=True)
					torch.save(raw_model.state_dict(), save_models_dir+"/{}-{:.3f}.pkl".format(strftime, accs.mean()))
					model.train()
				
			if not is_train:
				test_loss = float(np.mean(losses))
				logger.info("epoch, test loss: %d %f", epoch_num, test_loss)
				# print(test_loss)
				return test_loss

		best_return = -float('inf')

		self.tokens = 0 # counter used for learning rate decay
		
		# for testing 
		if self.config.is_eval_only:
			eval_return = {}
			T_scores_y_all_1 = []
			T_scores_y_all_err_1 = []
			T_scores_x_all_1 = []
			
			T_scores_x = []
			T_scores_y = []
			T_rewards_x_all_macro = []
			T_rewards_y_all_macro = []
			
			# plt.cla()
			tmp_y_1 = []
			for level in [0]:
				print("//------------------------------------------------")
				level /= 100.0
				actions = self.get_returns(level, self.test_dataset)
				"""actions here are hw positions of the workers"""
				retrieve_config(self.exp_config, actions, self.exp_config.save_idx)
				print("REFINE ACTION")
				
			print("=====================ALL DONE!=====================")
			return 

		for epoch in range(config.max_epochs):
			run_epoch('train', epoch_num=epoch)
			if (epoch + 1) % 400 == 0:
				if self.config.model_type == 'naive':
					assert False
				elif self.config.model_type == 'reward_conditioned':
					eval_return = {}
					for i, benchmark in enumerate(benchmark_list):
						if benchmark not in placedb_g_lib:
							continue
						if i > 0:
							break
						
						eval_return[benchmark] = {}
						T_scores_x = []
						T_scores_y = []
						# plt.cla()
						for level in [0]:
							level /= 100.0
							
							eval_return[benchmark][str(level)], T_scores = self.get_returns(level, benchmark = benchmark)
							
							for t in T_scores:
								T_scores_x.append(level)
								T_scores_y.append(t)
						strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
						# plt.savefig('./{}-{}-{}.png'.format(strftime, benchmark, epoch), 
						#     dpi=300, bbox_inches='tight', pad_inches = 0.1)
						# plt.cla()
						# self.writer.add_scalars('eval_{}'.format(benchmark), eval_return[benchmark], epoch)
				else:
					raise NotImplementedError()

	
	def get_returns(self, ret, test_dataset, is_shuffle_benchmark_id = False):
		
		loader = DataLoader(test_dataset, shuffle=True, pin_memory=True,
					batch_size=self.config.batch_size,
					num_workers=self.config.num_workers
					)
		
		# The recent observation: the very last observation
		pbar = enumerate(loader)
		for it, (x, y, r, t, m_x, b, st, cir, l) in pbar:   
			x = x.to(self.device)[0, -1, :]  # states: my=> (batch, context, 8*grid*grid)
			m_x = m_x.to(self.device)[0, -1, :]  # meta: my=> (batch, context, 6)
			y = y.to(self.device)[0, -1, :]  # action: my=> (batch, context, 1)
			r = r.to(self.device)[0, :, :, :]  # rtg: my=> (batch, context, 1, 1) should be (batch, context, 1)
			t = t.to(self.device)[0, -1, :]  # ts: my=> (batch, context, 1)
			b = b.to(self.device)[0, -1, :, :]  # benchmark: my=> (batch, context, 1, 1)
			st = st.to(self.device)[0, :, :, :]  # stepwise returns: my=> (batch, context, 1, 1)
			cir = cir.to(self.device)[0]  # circuit: my=> (batch, 768)
			l = l.to(self.device)[0]  # where to stop in a batch: my=> (batch, context)
		
		print(x.shape, y.shape, r.shape, t.shape, m_x.shape, b.shape, st.shape, cir.shape, l.shape)
		# zz=input()
		self.model.train(False)
		

		benchmark_id = torch.tensor(0, dtype=torch.int64).reshape(1, 1)
		T_rewards, T_Qs = [], []
		T_scores = []
		done = True


		#   circuit_feas_for_benchmark = torch.tensor(circuit_feas[benchmark_id], dtype = torch.float32)
		circuit_feas_for_benchmark = torch.randn(768)
		chassis_dimx = self.exp_config.chassis_dim[0]
		chassis_dimy = self.exp_config.chassis_dim[1]
		num_features = self.exp_config.num_features+1  # 1 to accound for grid index: 0, 1, 2, ...
		bound_core = int(self.exp_config.cnt_grid_cells / self.exp_config.machine.num_worker)+1
		
		print("// loop: repeat number ---------------------------------------------------------------------------")
		# The very first observation:
		# state, reward_sum, done, meta_state = env.reset()
		state_obs = torch.tensor(np.full((1, chassis_dimx, chassis_dimy), False))
		state_obs_s = torch.tensor(np.full((num_features, chassis_dimx, chassis_dimy), 0, dtype=np.float64))

		cores_position = self.exp_config.machine.worker_to_chassis_pos_mapping 
		
		state_obs_mask = np.full((chassis_dimx * chassis_dimy,), False)
		obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
		chassis_act_=[int(cores_position[int(z)]) for z in range(self.exp_config.machine.num_worker)]
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
		
		state = state.type(torch.float32).to(self.device).unsqueeze(0)
		meta_state = meta_state.type(torch.float32).to(self.device).unsqueeze(0)
		
		rtgs = [ret]  						# = [0]
		rtgs[0] = r.view(-1, )[0]  			# you set this yourself, hence it comes pre-packaged from the test set where it is set to max
		print("Desired Return = ", rtgs)
		print(state.shape)
		print(meta_state.shape)
		
		
		# ------------------------------------------------------------
		# sampled_action = (1, 1), action_probs = (1, vocab_size)
		sampled_action, action_probs = infer_action(
			self.model, state.unsqueeze(0), 1, self.exp_config, 
			temperature=1.0, sample=True, actions=None, 
			rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1), 
			timesteps = torch.arange(0, 2, dtype = torch.int64).reshape(1, 2, 1).to(self.device), 
			meta_state = meta_state, benchmarks = benchmark_id.to(self.device),
			stepwise_returns = None,
			circuit_feas = circuit_feas_for_benchmark.to(self.device),
			is_random_shuffle = is_shuffle_benchmark_id)
		
		
		j = 0
		all_states = state.type(torch.float32)
		all_meta_states = meta_state.type(torch.float32)
		actions = []
		print(all_states.shape, all_meta_states.shape)
		
		while True:
			if done:
				score_sum = 0
			
			action = sampled_action.cpu().numpy()[0,-1]
			if isinstance(action, int):
				actions += [action]
			else:
				actions += [action.item()]
			print(actions)
			
			if(len(actions) == self.exp_config.cnt_grid_cells): 
				# print(actions)
				return actions
			

			# ================
			# update the state and every other stuff now that you have seen the action
			# state, reward, done, meta_state = env.step(action)
			print("------------------------------------------------------------")
			
			# state, reward, done, meta_state = my_env_step(x, m_x, r, actions, self.assuming_cfg_idx)
			# state, reward, done, meta_state = my_env_step_new(x, m_x, st, actions, self.assuming_cfg_idx)
			reward = torch.tensor(rtgs)
			
			# zz = input()
			state, reward, done, meta_state, obs_mask_core = env_update(
				x, m_x, st, 
				actions, state, meta_state, reward, self.exp_config, 
				self.assuming_cfg_idx, obs_mask_core)
			
			# print(state.shape, reward.shape, meta_state.shape)
			# ours is the final format 
			
			# print(state.view(-1, 8, 8, 8)[:, 0, :, :])
			# print(meta_state.shape)
			# print(state.view(-1, 8, 8, 8)[:, 2:6, :, :])
			print("reward=", reward, "action=", action)
			# zz = input()
			
			# score = get_norm_reward(reward, benchmark, benchmark_to_id[benchmark], env.placed_num_macro)
			# reward_sum += reward
			# scores.append(score)
			# rewards.append(reward)
			# probs.append(action_probs[0, action].item())
			# score_sum += score
			j += 1

			if done:
				T_rewards.append(reward_sum)
				T_scores.append(score_sum)
				break

			# state = state.type(torch.float32).unsqueeze(0).to(self.device)
			# meta_state = meta_state.type(torch.float32).to(self.device).unsqueeze(0)
			# all_states = torch.cat([all_states, state], dim=0)
			# all_meta_states = torch.cat([all_meta_states, meta_state], dim=0)
			# rtgs += [rtgs[-1] - score]
			
			rtgs = reward.tolist()

			# print(state.shape)
			# print(reward.shape)
			# print(meta_state.shape)
			# print(all_states.shape)
			# print(all_meta_states.shape)
			# print(rtgs)

			sampled_action, action_probs = infer_action(
				self.model, state.unsqueeze(0), 1, 
				self.exp_config, 
				temperature=1.0, sample=True, 
				actions=torch.tensor(np.array(actions), dtype=torch.float32).to(self.device).unsqueeze(0),
				rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1), 
				timesteps = torch.arange(0, min(j+2, seq_len), dtype= torch.int64).reshape(1, -1, 1).to(self.device),
				meta_state = meta_state.unsqueeze(0), benchmarks = benchmark_id.to(self.device),
				stepwise_returns = None,
				circuit_feas = circuit_feas_for_benchmark.to(self.device),
				is_random_shuffle = is_shuffle_benchmark_id)
