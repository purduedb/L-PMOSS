import sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import copy
import os 
from pmoss_configs import processor_dict
def load_edge_index(cGridCell):
		if machine == 0:
				sample_array = np.loadtxt("/home/yrayhan/works/lpmoss/kb_b/" + str(CPUID[0]) + "/query_view.txt")
		else:
				sample_array = np.loadtxt("/home/yrayhan/works/lpmoss/kb_icelake_quad/" + str(CPUID[0]) + "/query_view.txt")
		edge_indexes = [[] for _ in range(sample_array.shape[0])]
		edge_indexes_w = [[] for _ in range(sample_array.shape[0])]

		for _ in CPUID:
				if machine == 0:
						RAW_FILE = "/home/yrayhan/works/lpmoss/kb_b/" + str(_) + "/query_view.txt"
				else:
						RAW_FILE = "/home/yrayhan/works/lpmoss/kb_icelake_quad/" + str(_) + "/query_view.txt"
				raw_array = np.loadtxt(RAW_FILE)
				idx_array = raw_array[:, 0:2]
				qCorr_array = np.reshape(raw_array[:, 2:], (raw_array.shape[0], cGridCell, cGridCell))

				for _ in range(qCorr_array.shape[0]):
						qCorr_ts = qCorr_array[_]  # ts
						for row in range(cGridCell):
								indexes = np.where(qCorr_ts[row] > 0)
								for n in indexes[0]:
										edge_indexes[_].append([row, n])
										edge_indexes_w[_].append(qCorr_ts[row, n])

		return edge_indexes, edge_indexes_w



def load_hardware_snapshot(exp_config):
		cfg_par = exp_config.cfg_par  
		sample_array = np.loadtxt(exp_config.kb_path + str(exp_config.machine.li_ncore_dumper[0]) + "/data_view.txt")
		
		GRID_FEATURES = np.zeros((sample_array.shape[0], exp_config.cnt_grid_cells, exp_config.num_features))
		GRID_QUERIES = np.zeros((sample_array.shape[0], exp_config.cnt_grid_cells))

		for _ in exp_config.machine.li_ncore_dumper:
				RAW_FILE = exp_config.kb_path+str(_) + "/data_view.txt"
				raw_array = np.loadtxt(RAW_FILE)

				cfgs_info = raw_array[:, 0:cfg_par]
				feature_plus_samples_array = raw_array[:, cfg_par:]

				feature_array = feature_plus_samples_array[:, :-exp_config.cnt_grid_cells]
				feature_array = np.reshape(feature_array, (feature_array.shape[0], exp_config.cnt_grid_cells, -1))
				feature_array = feature_array[:, :, :exp_config.num_features]
				
				GRID_FEATURES = np.add(GRID_FEATURES, feature_array)

				samples_array = feature_plus_samples_array[:, -exp_config.cnt_grid_cells:]
				GRID_QUERIES = np.add(GRID_QUERIES, samples_array)
				# samples_array = np.repeat(samples_array, exp_config.num_features, axis=1)
				# samples_array = np.reshape(samples_array, (-1, exp_config.cnt_grid_cells, exp_config.num_features))
				#
				# feature_array = np.divide(feature_array, samples_array)
				# feature_array = np.nan_to_num(feature_array, neginf=0, nan=0)

		grid_queries = np.repeat(GRID_QUERIES, exp_config.num_features, axis=1)
		grid_queries = np.reshape(grid_queries, (-1, exp_config.cnt_grid_cells, exp_config.num_features))

		GRID_FEATURES = np.nan_to_num(GRID_FEATURES, neginf=0, nan=0, posinf=9999999999)
		grid_features = np.divide(GRID_FEATURES, grid_queries, out=np.zeros_like(GRID_FEATURES), where=grid_queries!=0)
		
		# grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
		# grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
		# grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
		# grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))
		# grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)
		
		return cfgs_info, grid_features


def load_uncore_features_intel(exp_config):
		# Only for intel
		cfg_par = exp_config.cfg_par 
		num_numa = exp_config.machine.numa_node
		num_mc_per_numa = exp_config.machine.mc_channel_per_numa
		num_socket = exp_config.machine.socket
		num_upi_per_socket = exp_config.machine.num_upi_per_socket
		RAW_FILE = exp_config.kb_path + str(exp_config.machine.li_ncore_dumper[0]) + "/mem-channel_view.txt"
		raw_array = np.loadtxt(RAW_FILE)
		
		t1=cfg_par
		t2=cfg_par+num_numa*num_mc_per_numa
		read_channels = raw_array[:, t1:t2]
		t1=t2
		t2=cfg_par+2*num_numa*num_mc_per_numa
		write_channels = raw_array[:, t1:t2]
		t1=t2
		t2=cfg_par+2*num_numa*num_mc_per_numa+num_socket*num_upi_per_socket
		upi_channels_incoming = raw_array[:, t1:t2]
		t1=t2
		t2=cfg_par+2*num_numa*num_mc_per_numa+2*num_socket*num_upi_per_socket
		upi_channels_outgoing = raw_array[:, t1:t2]
		

		cfg = raw_array[:, :cfg_par]

		read_channels = np.reshape(read_channels, (read_channels.shape[0], num_numa, -1))
		write_channels = np.reshape(write_channels, (write_channels.shape[0], num_numa, -1))
		upi_incoming = np.reshape(upi_channels_incoming, (upi_channels_incoming.shape[0], num_socket, -1))
		upi_outgoing = np.reshape(upi_channels_outgoing, (upi_channels_outgoing.shape[0], num_socket, -1))

		read_channels_throughput_ts = np.sum(read_channels, axis=2)
		write_channels_throughput_ts = np.sum(write_channels, axis=2)
		upi_incoming_throughput_ts = np.sum(upi_incoming, axis=2)
		upi_outgoing_throughput_ts = np.sum(upi_outgoing, axis=2)


		read_channels_throughput_ts = np.reshape(read_channels_throughput_ts,
																						 (read_channels_throughput_ts.shape[0], num_numa, -1))
		write_channels_throughput_ts = np.reshape(write_channels_throughput_ts,
																							(write_channels_throughput_ts.shape[0], num_numa, -1))
		upi_incoming_throughput_ts = np.reshape(upi_incoming_throughput_ts,
																							(upi_incoming_throughput_ts.shape[0], num_socket, -1))
		upi_outgoing_throughput_ts = np.reshape(upi_outgoing_throughput_ts,
																							(upi_outgoing_throughput_ts.shape[0], num_socket, -1))

		return cfg, read_channels_throughput_ts, write_channels_throughput_ts, upi_incoming_throughput_ts, upi_outgoing_throughput_ts


def load_qtput_cum(exp_config):
		cfg_par = exp_config.cfg_par 
		grp_len = exp_config.per_cfg_sample

		raw_tL = []
		idx = []
		for _ in exp_config.machine.li_ncore_dumper:
				RAW_FILE = exp_config.kb_path + str(_) + "/query-exec_view.txt"
				raw_array = np.loadtxt(RAW_FILE, dtype=int)
				bw = np.sum(raw_array[:, cfg_par:], axis=1)
				raw_tL.append(bw)
				cfg = np.reshape(raw_array[:, 0:cfg_par], (raw_array[:, 0:1].shape[0], cfg_par))
				idx.append(cfg)

		query_throughput = np.asarray(raw_tL)
		query_throughput = np.moveaxis(query_throughput, 0, 1)
		query_throughput = query_throughput.sum(axis=1)
		
		query_throughput_shadow = np.zeros_like(query_throughput)
		query_throughput_shadow[1:] = query_throughput[:query_throughput.shape[0]-1]
		
		
		for _ in range(0, query_throughput_shadow.shape[0]):
				query_throughput_shadow[_] = query_throughput_shadow[0]
				_ = _ + grp_len
		query_throughput = query_throughput - query_throughput_shadow
		
		idx_array = np.asarray(idx)
		return idx_array, query_throughput


def find_correct_max_tput_for_wl(exp_config):
		cfg_q2, query_throughput_numa = load_qtput_cum(exp_config)  # (tr, )
		valid_tputs = []
		valid_idxs = []
		valid_cfgs = []
		
		
		for _ in range(query_throughput_numa.shape[0]):  # tr
				cfg_ = cfg_q2[0][_][0]
				wl_ = cfg_q2[0][_][2]
				if wl_ == exp_config.workload:
						valid_tputs.append(query_throughput_numa[_])
						valid_cfgs.append(cfg_)
						valid_idxs.append(_)
		
		tputs_to_consider = np.asarray(valid_tputs)
		max_tput = np.max(tputs_to_consider)
		mIdx = np.argmax(tputs_to_consider)
		
		maxCfg = valid_cfgs[mIdx]
		maxIdx = valid_idxs[mIdx]    
		print("MAXCONFIG =", maxCfg, "MAXIDX=", maxIdx)
		print(max_tput)
		return max_tput


def find_correct_max_tput(exp_config):
		cfg_q2, query_throughput_numa = load_qtput_cum(exp_config)  # (tr, )
		valid_tputs = []
		valid_idxs = []
		valid_cfgs = []
		
		
		for _ in range(query_throughput_numa.shape[0]):  # tr
				
				cfg_ = cfg_q2[0][_][0]
				
				# if cfg_ >=  1000:
				#     continue

				# comment for quadtree: aug 2
				# if _ <= 996 or 1677 <= _ <= 1707 or 1846 <= _ <= 1856 or 1917 <= _ <= 1931:
				#     continue
						
				valid_tputs.append(query_throughput_numa[_])
				valid_cfgs.append(cfg_)
				valid_idxs.append(_)
		
		tputs_to_consider = np.asarray(valid_tputs)
		max_tput = np.max(tputs_to_consider)
		mIdx = np.argmax(tputs_to_consider)
		
		maxCfg = valid_cfgs[mIdx]
		maxIdx = valid_idxs[mIdx]    
		print("MAXCONFIG =", maxCfg, "MAXIDX=", maxIdx)
		print(max_tput)
		return max_tput



def get_wkload_range(dataset, id, wkload, machine):
		ranges = []
		cfgs_to_choose= []
		if dataset == 'glife' and id == 0 and wkload == 0 and machine == '4S8N':
				ranges = [(1, 264), (841, 996), (1857, 1866), (1932, 1936), (1947, 1951)]
				# cfgs_to_choose = [900, 901, 30, 40, 1, 100601]
				cfgs_to_choose = [900, 901, 30, 40, 1, 200001]
				# --------------------------------------------------------------------------------------------
		elif dataset == 'glife' and id == 1 and wkload == 0 and machine == '4S8N':
				ranges = [(81, 155), (276, 285), (506, 590)]
				cfgs_to_choose = [900, 901, 30, 40, 20, 100601]  # [Uni-Scan] [QTree] [GLife] [BigData]
		# --------------------------------------------------------------------------------------------
		# --------------------------------------------------------------------------------------------
		elif dataset == 'bmod02' and id == 0 and wkload == 0 and machine == '4S8N':
				ranges = [(997, 1312), (1867, 1876), (1937, 1946), (1982, 1986)]
				# cfgs_to_choose = [900, 901, 30, 40, 1, 200001]
				cfgs_to_choose = [900, 901, 30, 40, 1, 300001]
				# --------------------------------------------------------------------------------------------
		elif dataset == 'bmod02' and id == 1 and wkload == 0 and machine == '4S8N':
				ranges = [(156, 220), (261, 275), (356, 360), (261, 275), (341, 355), (361, 370), (376, 505),
									]

				cfgs_to_choose = [900, 901, 30, 40, 1, 100601]
				cfgs_to_choose = [900, 901, 30, 40, 1, 400001]  # [Uni-Scan] [QTree] [BMod02] [BigData]
		# --------------------------------------------------------------------------------------------
		# -----------------------------[2 SOCKET SERVER][BMOD 02]---------------------------------------------------------------
		elif dataset == 'bmod02' and id == 0 and wkload == 9 and machine == '2S':
				ranges = [(141, 144)]
				cfgs_to_choose = [900, 901, 2000, 2010, 2020, 400001]
		elif dataset == 'bmod02' and id == 1 and wkload == 11 and machine == '2S':
				ranges = [(61, 80), (130, 139)]
				cfgs_to_choose = [1000, 1001, 2000, 2010, 2020, -1]  # [Uni-Scan] [QTree] [BMod02] [BigData]
		elif dataset == 'bmod02' and id == 1 and wkload == 12 and machine == '2S':
				ranges = [(81, 92), (140, 149)]
				cfgs_to_choose = [1000, 1001, 2000, 2010, 2020, -1]  # [Uniform] [Quad] [OSM] [DBServer]
		elif dataset == 'bmod02' and id == 1 and wkload == 13 and machine == '2S':
				ranges = [(93, 104), (150, 159)]
				cfgs_to_choose = [1000, 1001, 2000, 2010, 2020, -1]  # [Uniform] [Quad] [OSM] [DBServer]
		elif dataset == 'bmod02' and id == 1 and wkload == 14 and machine == '2S':
				ranges = [(105, 116), (160, 169)]
				cfgs_to_choose = [1000, 1001, 2000, 2010, 2020, -1]  # [Uniform] [Quad] [OSM] [DBServer]
		elif dataset == 'bmod02' and id == 1 and wkload == 15 and machine == '2S':
				ranges = [(117, 129), (170, 179)]
				cfgs_to_choose = [1000, 1001, 2000, 2010, 2020, -1]  # [Uniform] [Quad] [OSM] [DBServer]
		# --------------------------------------------------------------------------------------------
		# -----------------------------[2 SOCKET SERVER][OSM]---------------------------------------------------------------
		elif dataset == 'osm' and id == 1 and wkload == 0 and machine == '2S':
				ranges = [(1, 80)]
				cfgs_to_choose = [1000, 1001, 2000, 2010, 2020, -1]  # [Uniform] [Quad] [OSM] [DBServer]
		# --------------------------------------------------------------------------------------------
		# --------------------------------------------------------------------------------------------
		elif dataset == 'osm' and id == 0 and wkload == 0 and machine == '4S8N':
				ranges = [(1313, 1432), (1812, 1816),(1827, 1836)]
				cfgs_to_choose = [900, 901, 30, 40, 1, 200001]
		elif dataset == 'osm' and id == 0 and wkload == 0 and machine == '2S':
				ranges = [(1, 135)]
				cfgs_to_choose = [1000, 1001, 2001, 2015, 2021, -1]
		elif dataset == 'osm' and id == 0 and wkload == 0 and machine == '1S':
				ranges = [(1, 25)]
				cfgs_to_choose = [50, 51, 3000, 3020, 3027, -1]
				# --------------------------------------------------------------------------------------------
				# --------------------------------------------------------------------------------------------
		elif dataset == 'osm' and id == 0 and wkload == 1 and machine == '4S8N':
				ranges = [(1433, 1507), (1837, 1846), (1952, 1956)]
				cfgs_to_choose = [900, 901, 30, 40, 10, 300002]  # [Normal] [RTree] [OSM] [BigData]
				# cfgs_to_choose = [900, 901, 30, 40, 10, 100600]  # [Normal] [RTree] [OSM] [BigData]
		elif dataset == 'osm' and id == 0 and wkload == 2 and machine == '4S8N':
				ranges = [(1508, 1587)]
				cfgs_to_choose = [-1, -1, 30, 40, 1, 100600]  
		elif dataset == 'osm' and id == 0 and wkload == 3 and machine == '4S8N':
				ranges = [(1587, 1637)]
				cfgs_to_choose = [-1, -1, 30, 40, 1, 100600]  
		elif dataset == 'osm' and id == 0 and wkload == 10 and machine == '4S8N':
				ranges = [(1678, 1707), (1847, 1851), (1917, 1931), (1962, 1966)]
				# cfgs_to_choose = [900, 32, 30, 40, 1, 100600]  # [Log-Normal] [RTree] [OSM] [BigData]
				cfgs_to_choose = [900, 32, 30, 40, 2, 300003]  # [Log-Normal] [RTree] [OSM] [BigData]
		elif dataset == 'osm' and id == 0 and wkload == 5 and machine == '4S8N':
				ranges = [(1708, 1722), (1907, 1916), (1957, 1961)]
				cfgs_to_choose = [900, 901, 30, 40, 1, 300004]    # [Zipfian: skew = 40] [RTree] [OSM] [BigData]
				# cfgs_to_choose = [900, 901, 30, 40, 1, 200001]  # [Zipfian: skew = 40] [RTree] [OSM] [BigData]
		elif dataset == 'osm' and id == 0 and wkload == 6 and machine == '4S8N':
				ranges = [(1737, 1756), (1817, 1821), (1877, 1886), (1967, 1971)]
				# cfgs_to_choose = [900, 901, 30, 40, 1, 200001]
				cfgs_to_choose = [900, 901, 30, 40, 1, 300005]  # [Hot=3] [RTree] [OSM] [BigData]
		elif dataset == 'osm' and id == 0 and wkload == 7 and machine == '4S8N':
				ranges = [(1757, 1776), (1887, 1896), (1972, 1976)]
				cfgs_to_choose = [900, 901, 30, 40, 1, 100601]
				cfgs_to_choose = [900, 901, 30, 40, 1, 300006]  # [Hot=5] [RTree] [OSM] [BigData]
		elif dataset == 'osm' and id == 0 and wkload == 8 and machine == '4S8N':
				ranges = [(1777, 1796), (1897, 1906), (1977, 1981)]
				cfgs_to_choose = [900, 901, 30, 40, 1, 100600]
				cfgs_to_choose = [900, 901, 30, 40, 1, 300007]  # [Hot=7] [RTree] [OSM] [BigData]
		elif dataset == 'osm' and id == 0 and wkload == 9 and machine == '4S8N':
				ranges = [(1797, 1811), (1822, 1826)]
				cfgs_to_choose = [-1, -1, 30, 40, 1, 200001]
				cfgs_to_choose = [-1, -1, 30, 40, 1, 100600]
				# --------------------------------------------------------------------------------------------
		elif dataset == 'osm' and id == 1 and wkload == 0 and machine == '4S8N':
				ranges = [(1, 80), (286, 295), (296, 340), (371, 374), (591, 625)]
				cfgs_to_choose = [900, 901, 30, 40, 1, 200100]  # [Uniform-Scan] [Quad] [OSM] [BigData]
				# cfgs_to_choose = [900, 901, 30, 40, 1, 300999]
		elif dataset == 'osm' and id == 1 and wkload == 10 and machine == '4S8N':
				ranges = [(256, 260)]
				cfgs_to_choose = [900, 901, 30, 40, 1, 200100]  # [Uniform-Scan] [Quad] [OSM] [BigData]
		elif dataset == 'osm' and id == 1 and wkload == 0 and machine == '2S':
				ranges = [(1, 60)]
				cfgs_to_choose = [1000, 1001, 2000, 2010, 2020, -1]  # [Uniform] [Quad] [OSM] [DBServer]
		elif dataset == 'osm' and id == 1 and wkload == 0 and machine == '1S':
				ranges = [(1, 70)]
				cfgs_to_choose = [50, 51, 3000, -1, 3020, -1]  # [Uniform] [Quad] [OSM] [DBServer----1SS]
		elif dataset == 'ycsb' and id == 3 and wkload == 6 and machine == '4S8N':
				ranges = [(306, 555)]
				cfgs_to_choose = [50, 51, 3000, -1, 3020, -1]  # [Uniform] [Quad] [OSM] [DBServer----1SS]
		elif dataset == 'ycsb' and id == 3 and wkload == 7 and machine == '4S8N':
				ranges = [(556, 662)]
				cfgs_to_choose = [50, 51, 3000, -1, 3020, -1]  # [Uniform] [Quad] [OSM] [DBServer----1SS]
		return ranges


def load_qtput_per_kscell(exp_config):
		cfg_par = exp_config.cfg_par 
		grp_len = exp_config.per_cfg_sample

		raw_tL = []
		idx = []
		for _ in exp_config.machine.li_ncore_dumper:
				RAW_FILE = exp_config.kb_path + str(_) + "/query-exec_view.txt"
				raw_array = np.loadtxt(RAW_FILE, dtype=int)
				bw = raw_array[:, cfg_par:]
				raw_tL.append(bw)
				cfg = np.reshape(raw_array[:, 0:cfg_par], (raw_array[:, 0:1].shape[0], cfg_par))
				idx.append(cfg)


		query_throughput = np.asarray(raw_tL)
		query_throughput = np.moveaxis(query_throughput, 0, 1)
		query_throughput = query_throughput.sum(axis=1)
		
		query_throughput_shadow = np.zeros_like(query_throughput)
		query_throughput_shadow[1:, :] = query_throughput[:query_throughput.shape[0]-1, :]
		
		for _ in range(0, query_throughput_shadow.shape[0]):  
				query_throughput_shadow[_] = query_throughput_shadow[0]
				_ = _ + grp_len
		
		query_throughput = query_throughput - query_throughput_shadow
		idx_array = np.asarray(idx)
		return idx_array, query_throughput


def load_machine_adjacency():
		edge_indexes = []
		edge_indexes_w = []

		RAW_FILE = "/home/yrayhan/works/lpmoss/machine_configs/raw_config.txt"
		raw_array = np.loadtxt(RAW_FILE)
		for _ in range(raw_array.shape[0]):
				indexes = np.where(raw_array[_] > 0)
				for n in indexes[0]:
						edge_indexes.append([_, n])
						edge_indexes_w.append(raw_array[_, n])
		return edge_indexes, edge_indexes_w


def load_frozen_cell_embeddings():
		emb = np.loadtxt("predicted_in.features")
		return emb


def load_actions(exp_config, cfg, wl, onlyNUMA=False):
		
		# 1. this should come from the machine, which are the worker threads and how they should be converted to 0 - num of workers
		# 2. at the end there is this 10 * 10 division, you can change it to a square root division.
		
		gen_core_dict = {}
		coreIdx = 0

		for _ in exp_config.machine.li_worker:
				gen_core_dict[_] = coreIdx
				coreIdx += 1
				
		
		refined_configs = []
		cIdx = int(cfg)
		
		# TODO: Have the machine name and access name posssibly then add the config stuff
		if exp_config.index==0:
				if cIdx >= 200:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + ".txt"
						else:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + ".txt"
				else:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + ".txt"
						else:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + ".txt"
		elif exp_config.index==1:
				if cIdx >= 200:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + "_r.txt"
						else:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + "_r.txt"
				else:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + "_r.txt"
						else:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + "_r.txt"
		
		config = np.loadtxt(machine_config_path)
		
		
		numa_core_div = exp_config.policy_dim[0]
		config_numa = config[:numa_core_div, :]
		config_core = config[numa_core_div:, :]

		config_numa = np.reshape(config_numa, (config_numa.shape[0] * config_numa.shape[1],))
		config_core = np.reshape(config_core, (config_core.shape[0] * config_core.shape[1],))

		refined_config = np.empty_like(config_core)
		for _ in range(config_core.shape[0]):
				refined_config[_] = gen_core_dict[config_core[_]]

		if onlyNUMA:
				refined_configs.append(config_numa)
		else:
				refined_configs.append(refined_config)

		refined_configs = np.asarray(refined_configs)
		refined_configs = np.reshape(refined_config, (-1, ))
		# refined_configs_li = refined_config.tolist()
		
		return refined_configs


def load_actions_hw_pos(exp_config, cfg, wl, onlyNUMA=False):
		
		# 1. this should come from the machine, which are the worker threads and how they should be converted to 0 - num of workers
		# 2. at the end there is this 10 * 10 division, you can change it to a square root division.
		cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
		gen_core_dict = {}
		coreIdx = 0

		for _ in exp_config.machine.li_worker:
				gen_core_dict[_] = coreIdx
				coreIdx += 1
				
		
		refined_configs = []
		cIdx = int(cfg)
		
		# TODO: Have the machine name and access name posssibly then add the config stuff
		if exp_config.index==0:
				if cIdx >= 200:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + ".txt"
						else:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + ".txt"
				else:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + ".txt"
						else:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + ".txt"
		elif exp_config.index==1:
				if cIdx >= 200:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + "_r.txt"
						else:
								machine_config_path = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(int(wl)) + "/c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + "_r.txt"
				else:
						if exp_config.cnt_grid_cells == 100:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + "_r.txt"
						else:
								machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + "_r.txt"
		
		config = np.loadtxt(machine_config_path)
		
		
		numa_core_div = exp_config.policy_dim[0]
		config_numa = config[:numa_core_div, :]
		config_core = config[numa_core_div:, :]

		config_numa = np.reshape(config_numa, (config_numa.shape[0] * config_numa.shape[1],))
		config_core = np.reshape(config_core, (config_core.shape[0] * config_core.shape[1],))

		refined_config = np.empty_like(config_core)
		for _ in range(config_core.shape[0]):
				refined_config[_] = cores_position[int(gen_core_dict[config_core[_]])]

		if onlyNUMA:
				refined_configs.append(config_numa)
		else:
				refined_configs.append(refined_config)

		refined_configs = np.asarray(refined_configs)
		refined_configs = np.reshape(refined_config, (-1, ))
		# refined_configs_li = refined_config.tolist()
		
		return refined_configs



def retrieve_config(exp_config, out_actions, cfg_idx):
		hw_chassis = exp_config.machine.worker_to_chassis_pos_mapping
		hw_workers = exp_config.machine.li_worker
		num_numa = exp_config.machine.numa_node
		numa_nodes = []
		workers = []
		for a_ in out_actions:
				worker_index = hw_chassis.index(a_)
				workers.append(hw_workers[worker_index])
				numa_nodes.append(worker_index%num_numa)
		
		
		numa_nodes = np.asarray(numa_nodes)
		workers = np.asarray(workers)

		retrieved_configs_numa = np.reshape(numa_nodes, (exp_config.policy_dim[0], exp_config.policy_dim[1]))
		retrieved_configs_core = np.reshape(workers, (exp_config.policy_dim[0], exp_config.policy_dim[1]))
		retrieved_configs = np.vstack((retrieved_configs_numa, retrieved_configs_core))
		retrieved_configs = retrieved_configs.astype(int)
		save_cfg_dir = "./pmoss_machine_configs/" + exp_config.processor + "/" + str(exp_config.workload)
		os.makedirs(save_cfg_dir, exist_ok=True)
		if exp_config.index == 0:
				cfg_file = save_cfg_dir + "/c_" + str(cfg_idx) + "_" + str(exp_config.cnt_grid_cells) + ".txt"
		elif exp_config.index == 1:
				cfg_file = save_cfg_dir + "/c_" + str(cfg_idx) + "_" + str(exp_config.cnt_grid_cells) + "_r.txt"
		
		np.savetxt(cfg_file, retrieved_configs, fmt='%i')
		return retrieved_configs


def sth(cfg_idx):
		# cfg = np.loadtxt("machine_configs/config_" + str(cfg_idx) + ".txt")[10:]
		cfg = np.loadtxt("/home/yrayhan/works/lpmoss/machine_configs/config_" + str(cfg_idx) + ".txt")[10:]
		cfg = np.reshape(cfg, (-1, ))
		unique, counts = np.unique(cfg, return_counts=True)
		d = dict(zip(unique, counts))
		print(len(d))
		print(d)

# sth(221)
# sth(3000)
# =======================================================================================

# Commented it oct 13
# =======================================================================================
# if machine == 0:
#     cGridCell = 100
#     nFeatures = 5 + 1  # Changed it to 23 + 1, previous: 5 + 1   # change here 10 + 1
#     num_numa = 8
#     cCorePerNuma = 8
#     cMCChannel = 3
# else:
#     cGridCell = 100
#     nFeatures = 5 + 1  # Changed it to 23 + 1, previous: 5 + 1   # change here 10 + 1
#     num_numa = 2
#     cCorePerNuma = 40
#     cMCChannel = 12  # this is not sure



# =======================================================================================
# =======================================================================================
# =======================================================================================
# idx_array, grid_features, grid_queries = load_hardware_snapshot(cGridCell, nFeatures-1)
# # (tr, 2) (tr, nGridCells, nFeatures) (tr, 100, 5)
# edge_indexes, edge_indexes_w = load_edge_index(cGridCell)  # list(tr)
# cfg_s, mc_read_tput, mc_write_tput = load_uncore_features_intel(num_numa, cMCChannel)
# # (tr, 2) (tr, 8, 1) (tr, 8, 1)
# cfg_q, query_throughput = load_qtput_cum()  # (tr, )
# mc_tput = np.concatenate([mc_read_tput, mc_write_tput], axis=2)
# m_adj, m_adj_weight = load_machine_adjacency()


def gen_token_for_eval(exp_config):
		# (tr, 2) (tr, nGridCells, nFeatures) (tr, 100, 5)
		idx_array, grid_features = load_hardware_snapshot(exp_config)
		cfg_q, query_throughput = load_qtput_per_kscell(exp_config)  # (tr, cGridCell)
		cfg_q2, query_throughput_numa = load_qtput_cum(exp_config)  # (tr, )
		if not(exp_config.num_meta_features == 0):
				cfg_q3, read_channels_throughput_ts, write_channels_throughput_ts, upi_incoming_throughput_ts, upi_outgoing_throughput_ts = load_uncore_features_intel(exp_config)
				upi_tput = np.concatenate([upi_incoming_throughput_ts, upi_outgoing_throughput_ts], axis=2) 
				mc_tput = np.concatenate([read_channels_throughput_ts, write_channels_throughput_ts], axis=2)
				upi_tput = np.reshape(upi_tput, (upi_tput.shape[0], -1))
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))        
				mc_tput = np.concatenate([mc_tput, upi_tput], axis=1)
				

		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		
		"""
		For cleaning the data in amd processors, it's a bad practice but well
		"""
		if exp_config.processor == "amd_epyc7543_2s_2n" or exp_config.processor == "amd_epyc7543_2s_8n":
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
				refine = [244, 245, 246, 251]
				for _ in refine:
						grid_features[:, _, :] = 0.00001
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		""""""
		scaler = StandardScaler()
		grid_features = scaler.fit_transform(grid_features)
		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))

		grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
		grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
		grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
		grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))
		grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)

		if not(exp_config.num_meta_features == 0):
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))
				scaler_mc = StandardScaler()
				mc_tput = scaler_mc.fit_transform(mc_tput)
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], 1, -1))
				mc_tput = np.repeat(mc_tput, exp_config.cnt_grid_cells, axis=1)
		
		orginal_max_tput_dset = find_correct_max_tput(exp_config) * exp_config.rtg_scale
		orginal_max_tput_dset = find_correct_max_tput_for_wl(exp_config) * exp_config.rtg_scale

		obss = []
		obss_s = []
		obss_mask = []
		actions = []
		stepwise_returns = []
		rtgs = []
		done_idxs = []
		timesteps = []
		metas = []
		actions_ = []

		cores_position = exp_config.machine.worker_to_chassis_pos_mapping 

		num_numa = exp_config.machine.numa_node
		num_worker_per_numa = exp_config.machine.worker_per_numa
		lbound_core = 2 # 100 / 56
		lbound_numa = 13 # 100/8
		chassis_dimx = exp_config.chassis_dim[0]
		chassis_dimy = exp_config.chassis_dim[1]

		for _ in range(idx_array.shape[0]):  # tr
				cfg_ = idx_array[_][0]
				wl_ = idx_array[_][2]
				
				if cfg_!= exp_config.eval_start_cfg and wl_ != exp_config.workload:
						continue
				
				# Load the actions (how many for each complete row? = no of grid cells)
				act_ = load_actions(exp_config, cfg_, wl_)
				actions.append(act_)

				if not(exp_config.num_meta_features == 0):
						# Load the meta mc_data
						metas.append(mc_tput[_])

				
				"""Update the obss mask: mask should be where you should not put
						Which one to off? 
				"""
				numa_machine_obs = np.full((chassis_dimx * chassis_dimy, ), False)
				numa_machine_obs_s = np.full((chassis_dimx * chassis_dimy, exp_config.num_features+1), 0, dtype=np.float64)
				# numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), True)
				
				numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
				obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
				chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
				obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
				mask_already_full = np.where(obs_mask_core==0)
				numa_machine_obss_mask[mask_already_full] = True


				st_return = np.full((act_.shape[0], ), 0)
				tg_return = np.full((act_.shape[0]+1, ), 0) # => made it +1: april 16
				
				max_tput_dset = orginal_max_tput_dset
				tg_return[0] = max_tput_dset

				obss.append(copy.deepcopy(numa_machine_obs))  # empty obs
				obss_s.append(copy.deepcopy(numa_machine_obs_s))
				obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
				
				# print(torch.tensor(obss).view(-1, 8, 8))
				# print(query_throughput[_])
				# print(query_throughput_numa[_])

				flag = False 
				actions_individual = np.zeros_like(act_)
				for i in range(act_.shape[0]):  # each timestep 
						# map_idx=exp_config.machine.li_worker.index(int(act_[i]))
						# a = exp_config.machine.worker_to_chassis_pos_mapping[map_idx]
						a = int(cores_position[int(act_[i])])
						actions_individual[i] = a

						# => Add this if condition so that i can place stuff again
						# if i > 32:
						#     obs_mask_core = np.full((num_numa * num_worker_per_numa, ), 2)
						# if i > 60 and not(flag):  # For obs_mask_core = [2+1]
						#     obs_mask_core = np.full((num_numa * num_worker_per_numa, ), 1)
						#     flag = True

						# numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
						# mask_already_full = obs_mask_core.nonzero()
						# numa_machine_obss_mask[mask_already_full] = True

						"""Update the obss mask: mask should be where you should not put
								Which one to off? 
						"""
						numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
						obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
						chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
						obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
						mask_already_full = np.where(obs_mask_core==0)
						numa_machine_obss_mask[mask_already_full] = True

						st_return[i] = query_throughput[_][i]
						# tg_return[0] should be the max value and equal across
						if i != act_.shape[0]-1:
								tg_return[i+1] = max_tput_dset - query_throughput[_][i]
						max_tput_dset = tg_return[i+1]
						
						# By placing the [ith] grid cell, at the [a]th place in the machine
						# you take the grid feature of the ith cell currently,
						# i am just replacing the values, what happens if they have diff value or whether this is not possible at all
						# actions.append(a)
						numa_machine_obs[int(a)] = True
						numa_machine_obs_s[int(a)] += grid_features[_][i]
						
						if i != act_.shape[0]-1:
								obss.append(copy.deepcopy(numa_machine_obs))
								obss_s.append(copy.deepcopy(numa_machine_obs_s))
								obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
						
						# print(torch.tensor(obss).view(-1, 8, 8))
						# zz = input()
						# print(torch.tensor(numa_machine_obs).view(8, 8))
						# print(torch.tensor(numa_machine_obss_mask).view(8, 8))
						# print("=======================")
						# if i == 10:
						#     exit()
				# print(tg_return)
				# print(st_return)
				# print("=======================")
				# actions.append(-1)
				actions_.append(actions_individual)
				

				done_idxs.append((_+1) * i)
				rtgs.append(copy.deepcopy(tg_return[:exp_config.cnt_grid_cells]))  # parameterize it 
				stepwise_returns.append(copy.deepcopy(st_return))
				timesteps.append(np.reshape(np.arange(act_.shape[0]), (-1, )))

				# print(torch.tensor(rtgs).view(-1, 100))
				# print(torch.tensor(stepwise_returns).view(-1, 100))
				
		# actions = np.asarray(actions)  # (nsamples * ngridcells, 1)
		# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
		# obss = np.asarray(obss)
		# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, nFeatures)
		# obss_s = np.asarray(obss_s)
		# obss_mask = np.asarray(obss_mask)

		# stepwise_returns = np.asarray(stepwise_returns)  # (nSamples * nGridcells, 1)
		# rtgs = np.asarray(rtgs) # (nSamples * nGridcells, 1)
		# done_idxs = np.asarray(done_idxs)  # (nSamples, 1)
		# timesteps = np.asarray(timesteps) # (nsamples * ngridcells, 1)

		# print(actions.shape)
		# print(obss.shape)
		# print(obss_s.shape)
		# print(obss_mask.shape)
		# print(stepwise_returns.shape)
		# print(rtgs.shape)
		# print(done_idxs.shape)
		# print(timesteps.shape)
		
		"""Send the positions in the hardware, because this is what we actually do the prediction on"""
		actions_ = np.reshape(np.asarray(actions_), (-1, ))  # (nsamples * ngridcells, 1)
		actions = np.reshape(np.asarray(actions), (-1, ))  # (nsamples * ngridcells, 1)
		# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
		obss = np.reshape(np.asarray(obss), (-1, 1, chassis_dimx, chassis_dimy))
		# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, nFeatures)
		obss_s = np.reshape(np.asarray(obss_s), (-1, 1, chassis_dimx, chassis_dimy, exp_config.num_features+1))
		obss_mask = np.reshape(np.asarray(obss_mask), (-1, 1, chassis_dimx, chassis_dimy))

		stepwise_returns = np.reshape(np.asarray(stepwise_returns), (-1, 1))  # (nSamples * nGridcells, 1)
		rtgs = np.reshape(np.asarray(rtgs), (-1, 1))  # (nSamples * nGridcells, 1)
		done_idxs = np.reshape(np.asarray(done_idxs), (-1, ))  # (nSamples, 1)
		timesteps = np.reshape(np.asarray(timesteps), (-1, ))  # (nsamples * ngridcells, 1)

		rtgs = rtgs / exp_config.rtg_div

		print(actions.shape)
		print(obss.shape)
		print(obss_s.shape)
		print(obss_mask.shape)
		print(stepwise_returns.shape)
		print(rtgs.shape)
		print(done_idxs.shape)
		print(timesteps.shape)
		
		if not(exp_config.num_meta_features == 0):
				metas = np.reshape(np.asarray(metas), (obss_s.shape[0], -1))  # (nSamples * nGridcells, 1)
				# metas = np.zeros((obss_s.shape[0], 6))

		
		# This definitely needs work
		lengths = np.full((obss_s.shape[0], 1), exp_config.cnt_grid_cells)  # it should be actually the number of valid actions until which timestep
		benchmarks = np.zeros((obss_s.shape[0], 1))
		
		'''Sedning actions_ instead of actions'''
		return obss, obss_s, obss_mask, actions_, stepwise_returns, rtgs, done_idxs, timesteps, metas, lengths, benchmarks


def gen_token_for_eval_for_all(glb_exp_config):
	obss = []
	obss_s = []
	obss_mask = []
	actions = []
	stepwise_returns = []
	rtgs = []
	done_idxs = []
	timesteps = []
	metas = []
	actions_ = []
	for exp_config in glb_exp_config:
		# (tr, 2) (tr, nGridCells, nFeatures) (tr, 100, 5)
		idx_array, grid_features = load_hardware_snapshot(exp_config)
		cfg_q, query_throughput = load_qtput_per_kscell(exp_config)  # (tr, cGridCell)
		cfg_q2, query_throughput_numa = load_qtput_cum(exp_config)  # (tr, )
		if not(exp_config.num_meta_features == 0):
			if('intel' in exp_config.processor):
				cfg_q3, read_channels_throughput_ts, write_channels_throughput_ts, upi_incoming_throughput_ts, upi_outgoing_throughput_ts = load_uncore_features_intel(exp_config)
				upi_tput = np.concatenate([upi_incoming_throughput_ts, upi_outgoing_throughput_ts], axis=2) 
				mc_tput = np.concatenate([read_channels_throughput_ts, write_channels_throughput_ts], axis=2)
				upi_tput = np.reshape(upi_tput, (upi_tput.shape[0], -1))
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))        
				mc_tput = np.concatenate([mc_tput, upi_tput], axis=1)
			else:
				mc_tput = np.full((idx_array.shape[0], exp_config.num_meta_features), -1)

		if(mc_tput.shape[1] != exp_config.num_global_meta_features):
			padding = np.full((mc_tput.shape[0], exp_config.num_global_meta_features-mc_tput.shape[1]), -1)
			mc_tput = np.hstack((mc_tput, padding))		

		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		
		"""
		For cleaning the data in amd processors, it's a bad practice but well
		"""
		# if exp_config.processor == "amd_epyc7543_2s_2n" or exp_config.processor == "amd_epyc7543_2s_8n":
		# 		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
		# 		refine = [244, 245, 246, 251]
		# 		for _ in refine:
		# 				grid_features[:, _, :] = 0.00001
		# 		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		""""""
		scaler = StandardScaler()
		grid_features = scaler.fit_transform(grid_features)
		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))

		"""
		Add processor specific features
		"""
		# p_feat = [-1, -1]
		# for key in processor_dict:
		# 		if key in exp_config.processor:
		# 			p_feat[0] = processor_dict[key]    
		# 			break
		# p_feat[1] = exp_config.machine.numa_node

		# grid_features_p = np.full((grid_features.shape[0], exp_config.cnt_grid_cells, 2), p_feat)
		grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
		grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
		grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
		grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))
		grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)
		# grid_features = np.concatenate([grid_features, grid_features_idx, grid_features_p], axis=2)


		if not(exp_config.num_meta_features == 0):
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))
				scaler_mc = StandardScaler()
				mc_tput = scaler_mc.fit_transform(mc_tput)
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], 1, -1))
				p_feat = [-1, -1]
				for key in processor_dict:
						if key in exp_config.processor:
							p_feat[0] = processor_dict[key]    
							break
				p_feat[1] = exp_config.machine.numa_node
				grid_features_p = np.full((mc_tput.shape[0], 1, 2), p_feat)
				mc_tput = np.concatenate([mc_tput, grid_features_p], axis=2)
				mc_tput = np.repeat(mc_tput, exp_config.cnt_grid_cells, axis=1)
		
		orginal_max_tput_dset = find_correct_max_tput(exp_config) * exp_config.rtg_scale
		orginal_max_tput_dset = find_correct_max_tput_for_wl(exp_config) * exp_config.rtg_scale

		
		cores_position = exp_config.machine.worker_to_chassis_pos_mapping 

		num_numa = exp_config.machine.numa_node
		num_worker_per_numa = exp_config.machine.worker_per_numa
		lbound_core = 2 # 100 / 56
		lbound_numa = 13 # 100/8
		chassis_dimx = exp_config.chassis_dim[0]
		chassis_dimy = exp_config.chassis_dim[1]

		for _ in range(idx_array.shape[0]):  # tr
				cfg_ = idx_array[_][0]
				wl_ = idx_array[_][2]
				
				if cfg_!= exp_config.eval_start_cfg and wl_ != exp_config.workload:
						continue
				
				# Load the actions (how many for each complete row? = no of grid cells)
				act_ = load_actions(exp_config, cfg_, wl_)
				actions.append(act_)

				if not(exp_config.num_meta_features == 0):
						# Load the meta mc_data
						metas.append(mc_tput[_])

				
				"""Update the obss mask: mask should be where you should not put
						Which one to off? 
				"""
				numa_machine_obs = np.full((chassis_dimx * chassis_dimy, ), False)
				numa_machine_obs_s = np.full((chassis_dimx * chassis_dimy, exp_config.num_features+1), 0, dtype=np.float64)
				# numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), True)
				
				numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
				obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
				chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
				obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
				mask_already_full = np.where(obs_mask_core==0)
				numa_machine_obss_mask[mask_already_full] = True


				st_return = np.full((act_.shape[0], ), 0)
				tg_return = np.full((act_.shape[0]+1, ), 0) # => made it +1: april 16
				
				max_tput_dset = orginal_max_tput_dset
				tg_return[0] = max_tput_dset

				obss.append(copy.deepcopy(numa_machine_obs))  # empty obs
				obss_s.append(copy.deepcopy(numa_machine_obs_s))
				obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
				
				# print(torch.tensor(obss).view(-1, 8, 8))
				# print(query_throughput[_])
				# print(query_throughput_numa[_])

				flag = False 
				actions_individual = np.zeros_like(act_)
				for i in range(act_.shape[0]):  # each timestep 
						# map_idx=exp_config.machine.li_worker.index(int(act_[i]))
						# a = exp_config.machine.worker_to_chassis_pos_mapping[map_idx]
						a = int(cores_position[int(act_[i])])
						actions_individual[i] = a

						# => Add this if condition so that i can place stuff again
						# if i > 32:
						#     obs_mask_core = np.full((num_numa * num_worker_per_numa, ), 2)
						# if i > 60 and not(flag):  # For obs_mask_core = [2+1]
						#     obs_mask_core = np.full((num_numa * num_worker_per_numa, ), 1)
						#     flag = True

						# numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
						# mask_already_full = obs_mask_core.nonzero()
						# numa_machine_obss_mask[mask_already_full] = True

						"""Update the obss mask: mask should be where you should not put
								Which one to off? 
						"""
						numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
						obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
						chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
						obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
						mask_already_full = np.where(obs_mask_core==0)
						numa_machine_obss_mask[mask_already_full] = True

						st_return[i] = query_throughput[_][i]
						# tg_return[0] should be the max value and equal across
						if i != act_.shape[0]-1:
								tg_return[i+1] = max_tput_dset - query_throughput[_][i]
						max_tput_dset = tg_return[i+1]
						
						# By placing the [ith] grid cell, at the [a]th place in the machine
						# you take the grid feature of the ith cell currently,
						# i am just replacing the values, what happens if they have diff value or whether this is not possible at all
						# actions.append(a)
						numa_machine_obs[int(a)] = True
						numa_machine_obs_s[int(a)] += grid_features[_][i]
						
						if i != act_.shape[0]-1:
								obss.append(copy.deepcopy(numa_machine_obs))
								obss_s.append(copy.deepcopy(numa_machine_obs_s))
								obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
						
						# print(torch.tensor(obss).view(-1, 8, 8))
						# zz = input()
						# print(torch.tensor(numa_machine_obs).view(8, 8))
						# print(torch.tensor(numa_machine_obss_mask).view(8, 8))
						# print("=======================")
						# if i == 10:
						#     exit()
				# print(tg_return)
				# print(st_return)
				# print("=======================")
				# actions.append(-1)
				actions_.append(actions_individual)
				

				done_idxs.append((_+1) * i)
				rtgs.append(copy.deepcopy(tg_return[:exp_config.cnt_grid_cells]))  # parameterize it 
				stepwise_returns.append(copy.deepcopy(st_return))
				timesteps.append(np.reshape(np.arange(act_.shape[0]), (-1, )))

				# print(torch.tensor(rtgs).view(-1, 100))
				# print(torch.tensor(stepwise_returns).view(-1, 100))
				
	# actions = np.asarray(actions)  # (nsamples * ngridcells, 1)
	# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
	# obss = np.asarray(obss)
	# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, nFeatures)
	# obss_s = np.asarray(obss_s)
	# obss_mask = np.asarray(obss_mask)

	# stepwise_returns = np.asarray(stepwise_returns)  # (nSamples * nGridcells, 1)
	# rtgs = np.asarray(rtgs) # (nSamples * nGridcells, 1)
	# done_idxs = np.asarray(done_idxs)  # (nSamples, 1)
	# timesteps = np.asarray(timesteps) # (nsamples * ngridcells, 1)

	# print(actions.shape)
	# print(obss.shape)
	# print(obss_s.shape)
	# print(obss_mask.shape)
	# print(stepwise_returns.shape)
	# print(rtgs.shape)
	# print(done_idxs.shape)
	# print(timesteps.shape)
	
	"""Send the positions in the hardware, because this is what we actually do the prediction on"""
	actions_ = np.reshape(np.asarray(actions_), (-1, ))  # (nsamples * ngridcells, 1)
	actions = np.reshape(np.asarray(actions), (-1, ))  # (nsamples * ngridcells, 1)
	# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
	obss = np.reshape(np.asarray(obss), (-1, 1, chassis_dimx, chassis_dimy))
	# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, nFeatures)
	obss_s = np.reshape(np.asarray(obss_s), (-1, 1, chassis_dimx, chassis_dimy, exp_config.num_features+1))
	obss_mask = np.reshape(np.asarray(obss_mask), (-1, 1, chassis_dimx, chassis_dimy))

	stepwise_returns = np.reshape(np.asarray(stepwise_returns), (-1, 1))  # (nSamples * nGridcells, 1)
	rtgs = np.reshape(np.asarray(rtgs), (-1, 1))  # (nSamples * nGridcells, 1)
	done_idxs = np.reshape(np.asarray(done_idxs), (-1, ))  # (nSamples, 1)
	timesteps = np.reshape(np.asarray(timesteps), (-1, ))  # (nsamples * ngridcells, 1)

	rtgs = rtgs / exp_config.rtg_div

	print(actions.shape)
	print(obss.shape)
	print(obss_s.shape)
	print(obss_mask.shape)
	print(stepwise_returns.shape)
	print(rtgs.shape)
	print(done_idxs.shape)
	print(timesteps.shape)
	
	if not(exp_config.num_meta_features == 0):
			metas = np.reshape(np.asarray(metas), (obss_s.shape[0], -1))  # (nSamples * nGridcells, 1)
			# metas = np.zeros((obss_s.shape[0], 6))

	
	# This definitely needs work
	lengths = np.full((obss_s.shape[0], 1), exp_config.cnt_grid_cells)  # it should be actually the number of valid actions until which timestep
	benchmarks = np.zeros((obss_s.shape[0], 1))
	
	'''Sedning actions_ instead of actions'''
	return obss, obss_s, obss_mask, actions_, stepwise_returns, rtgs, done_idxs, timesteps, metas, lengths, benchmarks


def gen_token(exp_config):
		
		# (tr, 2) (tr, nGridCells, nFeatures) (tr, 100, 5)
		idx_array, grid_features = load_hardware_snapshot(exp_config) # hardware snapshot + grid index 
		cfg_q, query_throughput = load_qtput_per_kscell(exp_config) # (tr, cGridCell)
		cfg_q2, query_throughput_numa = load_qtput_cum(exp_config) # (tr, )
		 

		if not(exp_config.num_meta_features == 0):
				cfg_q3, read_channels_throughput_ts, write_channels_throughput_ts, upi_incoming_throughput_ts, upi_outgoing_throughput_ts = load_uncore_features_intel(exp_config)
				upi_tput = np.concatenate([upi_incoming_throughput_ts, upi_outgoing_throughput_ts], axis=2) 
				mc_tput = np.concatenate([read_channels_throughput_ts, write_channels_throughput_ts], axis=2)
				upi_tput = np.reshape(upi_tput, (upi_tput.shape[0], -1))
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))        
				mc_tput = np.concatenate([mc_tput, upi_tput], axis=1)


		
		# print(idx_array.shape, grid_features.shape)
		# print(cfg_q.shape, query_throughput.shape)
		# print(cfg_q2.shape, query_throughput_numa.shape)
		# print(cfg_q3.shape, read_channels_throughput_ts.shape, write_channels_throughput_ts.shape)
		# print(mc_tput.shape)
		# (1931, 2) (1931, 100, 6) (1931, 100, 5)
		# (8, 1931, 2) (1931, 100)
		# (8, 1931, 2) (1931,)
		# (1931, 2) (1931, 8, 1) (1931, 8, 1)
		# (1931, 8, 2)
		

		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))

		"""
		For cleaning the data in amd processors, it's a bad practice but well
		"""
		if exp_config.processor == "amd_epyc7543_2s_2n" or exp_config.processor == "amd_epyc7543_2s_8n":
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
				refine = [244, 245, 246, 251]
				for _ in refine:
						grid_features[:, _, :] = 0.00001
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		""""""
		scaler = StandardScaler()
		grid_features = scaler.fit_transform(grid_features)
		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))


		grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
		grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
		grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
		grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))
		grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)
		
		# for _ in range(grid_features.shape[1]):
		#     print(grid_features[0, _, :])
		#     zz=input()

		if not(exp_config.num_meta_features == 0):
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))
				scaler_mc = StandardScaler()
				mc_tput = scaler_mc.fit_transform(mc_tput)
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], 1, -1))
				mc_tput = np.repeat(mc_tput, exp_config.cnt_grid_cells, axis=1)
		
		orginal_max_tput_dset = find_correct_max_tput(exp_config) * 1.0
		
		obss = []
		obss_s = []
		obss_mask = []
		actions = []
		stepwise_returns = []
		rtgs = []
		done_idxs = []
		timesteps = []
		metas = []
		actions_ = []
		num_numa = exp_config.machine.numa_node
		num_worker_per_numa = exp_config.machine.worker_per_numa
		lbound_core = 6 # 100 / 56
		lbound_numa = 13 # 100/8
		chassis_dimx = exp_config.chassis_dim[0]
		chassis_dimy = exp_config.chassis_dim[1]

		# TODO: Return timesteps, resolve the last state thingy
		# _ = each full sample: we are breaking it into timesteps
		
		cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
		
		for _ in range(idx_array.shape[0]):  # tr
				cfg_ = idx_array[_][0]
				wl_ = idx_array[_][2]
				# print(cfg_)
				# Load the actions (how many for each complete row? = no of grid cells)
				act_ = load_actions(exp_config, cfg_, wl_)
				actions.append(act_)

				if not(exp_config.num_meta_features == 0):
						# Load the meta mc_data
						metas.append(mc_tput[_])

				
				# TODO: 
				# obs_mask_core = np.full((num_numa * num_worker_per_numa, ), lbound_core)
				# obs_mask_numa = np.full((num_numa,), lbound_numa)
				
				

				numa_machine_obs = np.full((chassis_dimx * chassis_dimy, ), False)
				numa_machine_obs_s = np.full((chassis_dimx * chassis_dimy, exp_config.num_features+1), 0, dtype=np.float64)
				"""Update the obss mask: mask should be where you should not put
						Which one to off? 
				"""
				# numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), True)  # it should be false
				
				# obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 3)
				# obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
				# obs_mask_core[np.array(act_).astype(int)] = 1
				
				numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
				obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
				chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
				obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
				mask_already_full = np.where(obs_mask_core==0)
				numa_machine_obss_mask[mask_already_full] = True
				

				st_return = np.full((act_.shape[0], ), 0)
				tg_return = np.full((act_.shape[0]+1, ), 0)
				
				# max_tput_dset = orginal_max_tput_dset
				max_tput_dset = query_throughput_numa[_]
				tg_return[0] = max_tput_dset
				
				obss.append(copy.deepcopy(numa_machine_obs))  # empty obs
				obss_s.append(copy.deepcopy(numa_machine_obs_s))
				obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
				
				
				flag = False 
				i=0
				actions_individual = np.zeros_like(act_)
				for i in range(act_.shape[0]):  # each timestep 
						# since we already refine it to 0 - 63, we do not need it
						# map_idx=exp_config.machine.li_worker.index(int(act_[i]))
						# a = exp_config.machine.worker_to_chassis_pos_mapping[map_idx]
						
						a = int(cores_position[int(act_[i])])
						actions_individual[i] = a
						
						"""Update the obss mask: mask should be where you should not put
						Which one to off? 
						"""
						numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
						obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
						chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
						obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
						mask_already_full = np.where(obs_mask_core==0)
						numa_machine_obss_mask[mask_already_full] = True

						st_return[i] = query_throughput[_][i]
						# tg_return[0] should be the max value and equal across
						if i != act_.shape[0]-1:
								tg_return[i+1] = max_tput_dset - query_throughput[_][i]
						
						max_tput_dset = tg_return[i+1]
						
						# By placing the [ith] grid cell, at the [a]th place in the machine
						# you take the grid feature of the ith cell currently,
						# i am just replacing the values, what happens if they have diff value or whether this is not possible at all
						# actions.append(a)
						numa_machine_obs[int(a)] = True
						numa_machine_obs_s[int(a)] += grid_features[_][i]
						
						if i != act_.shape[0]-1:
								obss.append(copy.deepcopy(numa_machine_obs))
								obss_s.append(copy.deepcopy(numa_machine_obs_s))
								obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
						
						# print(numa_machine_obs)
						# print(numa_machine_obs_s)
						# print(numa_machine_obss_mask)
						# zz = input()

						# print(torch.tensor(obss).view(-1, 8, 8))
						# zz = input()
						# print(torch.tensor(numa_machine_obs).view(8, 8))
						# print(torch.tensor(numa_machine_obss_mask).view(8, 8))
						# print("=======================")
						# if i == 10:
						#     exit()
				# print(tg_return)
				# print(st_return)
				# print("=======================")
				# actions.append(-1)
				actions_.append(actions_individual)
				
				done_idxs.append((_+1) * i)
				rtgs.append(copy.deepcopy(tg_return[:exp_config.cnt_grid_cells]))  # parameterize it 
				stepwise_returns.append(copy.deepcopy(st_return))
				timesteps.append(np.reshape(np.arange(act_.shape[0]), (-1, )))

				# print(torch.tensor(rtgs).view(-1, 100))
				# print(torch.tensor(stepwise_returns).view(-1, 100))
				
		# actions = np.asarray(actions)  # (nsamples * ngridcells, 1)
		# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
		# obss = np.asarray(obss)
		# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, exp_config.num_features+1)
		# obss_s = np.asarray(obss_s)
		# obss_mask = np.asarray(obss_mask)

		# stepwise_returns = np.asarray(stepwise_returns)  # (nSamples * nGridcells, 1)
		# rtgs = np.asarray(rtgs) # (nSamples * nGridcells, 1)
		# done_idxs = np.asarray(done_idxs)  # (nSamples, 1)
		# timesteps = np.asarray(timesteps) # (nsamples * ngridcells, 1)

		# print(actions.shape)
		# print(obss.shape)
		# print(obss_s.shape)
		# print(obss_mask.shape)
		# print(stepwise_returns.shape)
		# print(rtgs.shape)
		# print(done_idxs.shape)
		# print(timesteps.shape)
		
		"""Send the positions in the hardware, because this is what we actually do the prediction on"""
		actions_ = np.reshape(np.asarray(actions_), (-1, ))  # (nsamples * ngridcells, 1)
		actions = np.reshape(np.asarray(actions), (-1, ))  # (nsamples * ngridcells, 1)
		# print(actions_.shape, actions.shape)
		# zz = input()

		# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
		obss = np.reshape(np.asarray(obss), (-1, 1, chassis_dimx, chassis_dimy))
		# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, exp_config.num_features)
		obss_s = np.reshape(np.asarray(obss_s), (-1, 1, chassis_dimx, chassis_dimy, exp_config.num_features+1))
		obss_mask = np.reshape(np.asarray(obss_mask), (-1, 1, chassis_dimx, chassis_dimy))

		stepwise_returns = np.reshape(np.asarray(stepwise_returns), (-1, 1))  # (nSamples * nGridcells, 1)
		rtgs = np.reshape(np.asarray(rtgs), (-1, 1))  # (nSamples * nGridcells, 1)
		done_idxs = np.reshape(np.asarray(done_idxs), (-1, ))  # (nSamples, 1)
		timesteps = np.reshape(np.asarray(timesteps), (-1, ))  # (nsamples * ngridcells, 1)

		
		rtgs = rtgs / exp_config.rtg_div
		
		
		print(actions.shape)
		print(obss.shape)
		print(obss_s.shape)
		print(obss_mask.shape)
		print(stepwise_returns.shape)
		print(rtgs.shape)
		print(done_idxs.shape)
		print(timesteps.shape)
		
		if not(exp_config.num_meta_features == 0):
				metas = np.reshape(np.asarray(metas), (obss_s.shape[0], -1))  # (nSamples * nGridcells, 1)
				# metas = np.zeros((obss_s.shape[0], 6))

		
		# This definitely needs work
		lengths = np.full((obss_s.shape[0], 1), exp_config.cnt_grid_cells)  # it should be actually the number of valid actions until which timestep
		benchmarks = np.zeros((obss_s.shape[0], 1))
		
		'''Sedning actions_ instead of actions'''
		return obss, obss_s, obss_mask, actions_, stepwise_returns, rtgs, done_idxs, timesteps, metas, lengths, benchmarks


def gen_token_for_all(glb_exp_config):
	obss = []
	obss_s = []
	obss_mask = []
	actions = []
	stepwise_returns = []
	rtgs = []
	done_idxs = []
	timesteps = []
	metas = []
	actions_ = []
	for exp_config in glb_exp_config:		
		# (tr, 2) (tr, nGridCells, nFeatures) (tr, 100, 5)
		idx_array, grid_features = load_hardware_snapshot(exp_config) # hardware snapshot + grid index 
		cfg_q, query_throughput = load_qtput_per_kscell(exp_config) # (tr, cGridCell)
		cfg_q2, query_throughput_numa = load_qtput_cum(exp_config) # (tr, )
		

		if not(exp_config.num_meta_features == 0):
			if('intel' in exp_config.processor):
				cfg_q3, read_channels_throughput_ts, write_channels_throughput_ts, upi_incoming_throughput_ts, upi_outgoing_throughput_ts = load_uncore_features_intel(exp_config)
				upi_tput = np.concatenate([upi_incoming_throughput_ts, upi_outgoing_throughput_ts], axis=2) 
				mc_tput = np.concatenate([read_channels_throughput_ts, write_channels_throughput_ts], axis=2)
				upi_tput = np.reshape(upi_tput, (upi_tput.shape[0], -1))
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))        
				mc_tput = np.concatenate([mc_tput, upi_tput], axis=1)
			else:
				mc_tput = np.full((idx_array.shape[0], exp_config.num_meta_features), -1)
		if(mc_tput.shape[1] != exp_config.num_global_meta_features):
			padding = np.full((mc_tput.shape[0], exp_config.num_global_meta_features-mc_tput.shape[1]), -1)
			mc_tput = np.hstack((mc_tput, padding))
			
			
			
		# print(idx_array.shape, grid_features.shape)
		# print(cfg_q.shape, query_throughput.shape)
		# print(cfg_q2.shape, query_throughput_numa.shape)
		# print(cfg_q3.shape, read_channels_throughput_ts.shape, write_channels_throughput_ts.shape)
		# print(mc_tput.shape)
		# (1931, 2) (1931, 100, 6) (1931, 100, 5)
		# (8, 1931, 2) (1931, 100)
		# (8, 1931, 2) (1931,)
		# (1931, 2) (1931, 8, 1) (1931, 8, 1)
		# (1931, 8, 2)
		
		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))

		"""
		For cleaning the data in amd processors, it's a bad practice but well
		"""
		# if exp_config.processor == "amd_epyc7543_2s_2n" or exp_config.processor == "amd_epyc7543_2s_8n":
		# 		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
		# 		refine = [244, 245, 246, 251]
		# 		for _ in refine:
		# 				grid_features[:, _, :] = 0.00001
		# 		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		""""""
		scaler = StandardScaler()
		grid_features = scaler.fit_transform(grid_features)
		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))


		"""
		Add processor specific features
		"""
		grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
		grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
		grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
		grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))
		grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)
		# grid_features = np.concatenate([grid_features, grid_features_idx, grid_features_p], axis=2)

		# for _ in range(grid_features.shape[1]):
		#     print(grid_features[0, _, :])
		#     zz=input()

		if not(exp_config.num_meta_features == 0):
			mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))
			scaler_mc = StandardScaler()
			mc_tput = scaler_mc.fit_transform(mc_tput)
			mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], 1, -1))
			p_feat = [-1, -1]
			for key in processor_dict:
				if key in exp_config.processor:
					p_feat[0] = processor_dict[key]    
					break
			p_feat[1] = exp_config.machine.numa_node
			grid_features_p = np.full((mc_tput.shape[0], 1, 2), p_feat)
			mc_tput = np.concatenate([mc_tput, grid_features_p], axis=2)
			mc_tput = np.repeat(mc_tput, exp_config.cnt_grid_cells, axis=1)
			mc_tput = np.repeat(mc_tput, exp_config.cnt_grid_cells, axis=1)
			

		orginal_max_tput_dset = find_correct_max_tput(exp_config) * 1.0
		
		# obss = []
		# obss_s = []
		# obss_mask = []
		# actions = []
		# stepwise_returns = []
		# rtgs = []
		# done_idxs = []
		# timesteps = []
		# metas = []
		# actions_ = []
		num_numa = exp_config.machine.numa_node
		num_worker_per_numa = exp_config.machine.worker_per_numa
		lbound_core = 6 # 100 / 56
		lbound_numa = 13 # 100/8
		chassis_dimx = exp_config.chassis_dim[0]
		chassis_dimy = exp_config.chassis_dim[1]

		# TODO: Return timesteps, resolve the last state thingy
		# _ = each full sample: we are breaking it into timesteps
		
		cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
		
		for _ in range(idx_array.shape[0]):  # tr
			cfg_ = idx_array[_][0]
			wl_ = idx_array[_][2]
			# print(cfg_)
			# Load the actions (how many for each complete row? = no of grid cells)
			act_ = load_actions(exp_config, cfg_, wl_)
			actions.append(act_)

			if not(exp_config.num_meta_features == 0):
					# Load the meta mc_data
					metas.append(mc_tput[_])
					
			
			# TODO: 
			# obs_mask_core = np.full((num_numa * num_worker_per_numa, ), lbound_core)
			# obs_mask_numa = np.full((num_numa,), lbound_numa)
			
			

			numa_machine_obs = np.full((chassis_dimx * chassis_dimy, ), False)
			numa_machine_obs_s = np.full((chassis_dimx * chassis_dimy, exp_config.num_features+1), 0, dtype=np.float64)
			"""Update the obss mask: mask should be where you should not put
					Which one to off? 
			"""
			# numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), True)  # it should be false
			
			# obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 3)
			# obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
			# obs_mask_core[np.array(act_).astype(int)] = 1
			
			numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
			obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
			chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
			obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
			mask_already_full = np.where(obs_mask_core==0)
			numa_machine_obss_mask[mask_already_full] = True
			

			st_return = np.full((act_.shape[0], ), 0)
			tg_return = np.full((act_.shape[0]+1, ), 0)
			
			# max_tput_dset = orginal_max_tput_dset
			max_tput_dset = query_throughput_numa[_]
			tg_return[0] = max_tput_dset
			
			obss.append(copy.deepcopy(numa_machine_obs))  # empty obs
			obss_s.append(copy.deepcopy(numa_machine_obs_s))
			obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
			
			
			flag = False 
			i=0
			actions_individual = np.zeros_like(act_)
			for i in range(act_.shape[0]):  # each timestep 
					# since we already refine it to 0 - 63, we do not need it
					# map_idx=exp_config.machine.li_worker.index(int(act_[i]))
					# a = exp_config.machine.worker_to_chassis_pos_mapping[map_idx]
					
					a = int(cores_position[int(act_[i])])
					actions_individual[i] = a
					
					"""Update the obss mask: mask should be where you should not put
					Which one to off? 
					"""
					numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
					obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
					chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
					obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
					mask_already_full = np.where(obs_mask_core==0)
					numa_machine_obss_mask[mask_already_full] = True

					st_return[i] = query_throughput[_][i]
					# tg_return[0] should be the max value and equal across
					if i != act_.shape[0]-1:
							tg_return[i+1] = max_tput_dset - query_throughput[_][i]
					
					max_tput_dset = tg_return[i+1]
					
					# By placing the [ith] grid cell, at the [a]th place in the machine
					# you take the grid feature of the ith cell currently,
					# i am just replacing the values, what happens if they have diff value or whether this is not possible at all
					# actions.append(a)
					numa_machine_obs[int(a)] = True
					numa_machine_obs_s[int(a)] += grid_features[_][i]
					
					if i != act_.shape[0]-1:
							obss.append(copy.deepcopy(numa_machine_obs))
							obss_s.append(copy.deepcopy(numa_machine_obs_s))
							obss_mask.append(copy.deepcopy(numa_machine_obss_mask))
					
					# print(numa_machine_obs)
					# print(numa_machine_obs_s)
					# print(numa_machine_obss_mask)
					# zz = input()

					# print(torch.tensor(obss).view(-1, 8, 8))
					# zz = input()
					# print(torch.tensor(numa_machine_obs).view(8, 8))
					# print(torch.tensor(numa_machine_obss_mask).view(8, 8))
					# print("=======================")
					# if i == 10:
					#     exit()
			# print(tg_return)
			# print(st_return)
			# print("=======================")
			# actions.append(-1)
			actions_.append(actions_individual)
			
			done_idxs.append((_+1) * i)
			rtgs.append(copy.deepcopy(tg_return[:exp_config.cnt_grid_cells]))  # parameterize it 
			stepwise_returns.append(copy.deepcopy(st_return))
			timesteps.append(np.reshape(np.arange(act_.shape[0]), (-1, )))

			# print(torch.tensor(rtgs).view(-1, 100))
			# print(torch.tensor(stepwise_returns).view(-1, 100))
				
	# actions = np.asarray(actions)  # (nsamples * ngridcells, 1)
	# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
	# obss = np.asarray(obss)
	# # (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, exp_config.num_features+1)
	# obss_s = np.asarray(obss_s)
	# obss_mask = np.asarray(obss_mask)

	# stepwise_returns = np.asarray(stepwise_returns)  # (nSamples * nGridcells, 1)
	# rtgs = np.asarray(rtgs) # (nSamples * nGridcells, 1)
	# done_idxs = np.asarray(done_idxs)  # (nSamples, 1)
	# timesteps = np.asarray(timesteps) # (nsamples * ngridcells, 1)

	# print(actions.shape)
	# print(obss.shape)
	# print(obss_s.shape)
	# print(obss_mask.shape)
	# print(stepwise_returns.shape)
	# print(rtgs.shape)
	# print(done_idxs.shape)
	# print(timesteps.shape)
	
	"""Send the positions in the hardware, because this is what we actually do the prediction on"""
	actions_ = np.reshape(np.asarray(actions_), (-1, ))  # (nsamples * ngridcells, 1)
	actions = np.reshape(np.asarray(actions), (-1, ))  # (nsamples * ngridcells, 1)
	# print(actions_.shape, actions.shape)
	# zz = input()

	# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa,num_worker_per_numa)
	obss = np.reshape(np.asarray(obss), (-1, 1, chassis_dimx, chassis_dimy))
	# (nSamples * (nGridcells+1 = initial observation = 0s), num_numa, num_worker_per_numa, exp_config.num_features)
	obss_s = np.reshape(np.asarray(obss_s), (-1, 1, chassis_dimx, chassis_dimy, exp_config.num_features+1))
	obss_mask = np.reshape(np.asarray(obss_mask), (-1, 1, chassis_dimx, chassis_dimy))

	stepwise_returns = np.reshape(np.asarray(stepwise_returns), (-1, 1))  # (nSamples * nGridcells, 1)
	rtgs = np.reshape(np.asarray(rtgs), (-1, 1))  # (nSamples * nGridcells, 1)
	done_idxs = np.reshape(np.asarray(done_idxs), (-1, ))  # (nSamples, 1)
	timesteps = np.reshape(np.asarray(timesteps), (-1, ))  # (nsamples * ngridcells, 1)

	
	rtgs = rtgs / exp_config.rtg_div
	
	
	print(actions.shape)
	print(obss.shape)
	print(obss_s.shape)
	print(obss_mask.shape)
	print(stepwise_returns.shape)
	print(rtgs.shape)
	print(done_idxs.shape)
	print(timesteps.shape)
	
	if not(exp_config.num_meta_features == 0):
		metas = np.reshape(np.asarray(metas), (obss_s.shape[0], -1))  # (nSamples * nGridcells, 1)
		# metas = np.zeros((obss_s.shape[0], 6))
	


	# This definitely needs work
	lengths = np.full((obss_s.shape[0], 1), exp_config.cnt_grid_cells)  # it should be actually the number of valid actions until which timestep
	benchmarks = np.zeros((obss_s.shape[0], 1))
	
	'''Sedning actions_ instead of actions'''
	return obss, obss_s, obss_mask, actions_, stepwise_returns, rtgs, done_idxs, timesteps, metas, lengths, benchmarks


def get_state(action, ts, exp_config):
		idx_array, grid_features = load_hardware_snapshot(exp_config)
		cfg_q, query_throughput = load_qtput_per_kscell(exp_config)  # (tr, cGridCell)
		 
		refined_idx = np.where(idx_array[:, 2] == exp_config.workload)[0]
		idx_array = idx_array[refined_idx]
		grid_features = grid_features[refined_idx]
		query_throughput = query_throughput[refined_idx]

		"""
		For cleaning the data in amd processors, it's a bad practice but well
		"""
		if exp_config.processor == "amd_epyc7543_2s_2n" or exp_config.processor == "amd_epyc7543_2s_8n":
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
				refine = [244, 245, 246, 251]
				for _ in refine:
						grid_features[:, _, :] = 0.00001
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		""""""

		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		scaler = StandardScaler()
		grid_features = scaler.fit_transform(grid_features)
		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
		
		grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
		grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
		grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
		grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))
		grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)
		
		actions = []
		for _ in range(idx_array.shape[0]):
				"""These are hardware positions"""
				a_ = load_actions_hw_pos(exp_config, idx_array[_][0], idx_array[_][2])
				actions.append(a_)
		
		actions = np.asarray(actions)
		
		eval_actions = actions[:, ts]

		ret_idx = np.where(eval_actions == action)    
		if ret_idx[0].shape[0] == 0:
				return torch.tensor([]), torch.tensor([])

		ret_state = grid_features[ret_idx[0]][:, ts, :]
		ret_state = np.average(ret_state, axis=0)
		ret_state = torch.tensor(ret_state)
		
		ret_query_tput = query_throughput[ret_idx[0]][:, ts]
		ret_query_tput = np.average(ret_query_tput, axis=0).item()
		
		# print(ret_query_tput)
		
		# ret_query_tput = torch.tensor(ret_query_tput)
		
		return ret_state, ret_query_tput


def get_state_up(action, ts, exp_config):
		idx_array, grid_features = load_hardware_snapshot(exp_config)
		cfg_q, query_throughput = load_qtput_per_kscell(exp_config)  # (tr, cGridCell)
		if not(exp_config.num_meta_features == 0):
				cfg_q3, read_channels_throughput_ts, write_channels_throughput_ts, upi_incoming_throughput_ts, upi_outgoing_throughput_ts = load_uncore_features_intel(exp_config)
				upi_tput = np.concatenate([upi_incoming_throughput_ts, upi_outgoing_throughput_ts], axis=2) 
				mc_tput = np.concatenate([read_channels_throughput_ts, write_channels_throughput_ts], axis=2)
				upi_tput = np.reshape(upi_tput, (upi_tput.shape[0], -1))
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))        
				mc_tput = np.concatenate([mc_tput, upi_tput], axis=1)


		"""
		For cleaning the data in amd processors, it's a bad practice but well
		"""
		if exp_config.processor == "amd_epyc7543_2s_2n" or exp_config.processor == "amd_epyc7543_2s_8n":
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
				refine = [244, 245, 246, 251]
				for _ in refine:
						grid_features[:, _, :] = 0.00001
				grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		""""""

		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
		scaler = StandardScaler()
		grid_features = scaler.fit_transform(grid_features)
		grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features))
		
		grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
		grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
		grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
		grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))
		grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)

		if not(exp_config.num_meta_features == 0):
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))
				scaler_mc = StandardScaler()
				mc_tput = scaler_mc.fit_transform(mc_tput)
				mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], 1, -1))
				mc_tput = np.repeat(mc_tput, exp_config.cnt_grid_cells, axis=1)


		refined_idx = np.where(idx_array[:, 2] == exp_config.workload)[0]
		idx_array = idx_array[refined_idx]
		grid_features = grid_features[refined_idx]
		query_throughput = query_throughput[refined_idx]
		mc_tput = mc_tput[refined_idx]
		
		actions = []
		for _ in range(idx_array.shape[0]):
				"""These are hardware positions"""
				a_ = load_actions_hw_pos(exp_config, idx_array[_][0], idx_array[_][2])
				actions.append(a_)
		
		actions = np.asarray(actions)
		
		eval_actions = actions[:, ts]

		ret_idx = np.where(eval_actions == action)    
		if ret_idx[0].shape[0] == 0:
				return torch.tensor([]), torch.tensor([]), torch.tensor([])

		ret_state = grid_features[ret_idx[0]][:, ts, :]
		ret_state = np.average(ret_state, axis=0)
		ret_state = torch.tensor(ret_state)
		
		ret_query_tput = query_throughput[ret_idx[0]][:, ts]
		ret_query_tput = np.average(ret_query_tput, axis=0).item()
		
		ret_meta_state = mc_tput[ret_idx[0]][:, ts, :]
		ret_meta_state = np.average(ret_meta_state, axis=0)
		ret_meta_state = torch.tensor(ret_meta_state, dtype=torch.float32)
		print(ret_meta_state.dtype)
		# print(ret_query_tput)
		
		# ret_query_tput = torch.tensor(ret_query_tput)
		print(ret_state.shape, ret_meta_state.shape)
		
		return ret_state, ret_query_tput, ret_meta_state


def env_update(
				x, m_x, st_return, 
				actions, current_x, current_mx, current_rtg, exp_config, 
				cfgIdx, obs_mask_core
				):
		
		"""
		TODO: This is what the model is predicting 
		so the actions are core indexes ]
		or the actions are machine board indexes 
		for now: machine board indexes
		now, stepwise-return is exactly the same as we have for the heuristics
		"""

		# current_rtg = current_rtg * 100000
		current_rtg = current_rtg * exp_config.rtg_div
		chassis_dimx = exp_config.chassis_dim[0]
		chassis_dimy = exp_config.chassis_dim[1]
		num_features = exp_config.num_features
		
		
		# These are the recent observations, 
		seen_obs = x.view(-1, chassis_dimx, chassis_dimy)
		seen_obss_s = seen_obs[1:num_features+2, :, :].view(-1, chassis_dimx*chassis_dimy)
		
		cfg_ = exp_config.eval_start_cfg
		cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
		"""these are from 0-numworkers"""
		seen_act_ = load_actions(exp_config, cfg_, exp_config.workload)
		"""These are hardware positions"""
		seen_a_= int(cores_position[int(seen_act_[-1])])
		
		
		rtgs = []
		metas = []
		
		# Load the meta mc_data
		# metas.append(mc_tput[_])

		# What you have accumulated so far
		numa_machine_obs = current_x[-1][0, :, :].view(-1, )
		numa_machine_obs_s = current_x[-1][1:num_features+2, :, :].view(-1, chassis_dimx*chassis_dimy)
		numa_machine_obss_mask = current_x[-1][num_features+2, :, :].view(-1, )
		tg_return = torch.tensor([0])
		# print(numa_machine_obs.shape, numa_machine_obs_s.shape, numa_machine_obss_mask.shape)
		# print(numa_machine_obs.view(chassis_dimx, chassis_dimy))
		# print(numa_machine_obs_s.view(exp_config.num_features+1, chassis_dimx, chassis_dimy)[-1])
		print(obs_mask_core)
		# print(numa_machine_obss_mask.view(chassis_dimx, chassis_dimy))
		# zz = input()

		cfg_q2, query_throughput_numa = load_qtput_cum(exp_config)  # (tr, )
		original_max_tput_dset = find_correct_max_tput_for_wl(exp_config) * exp_config.rtg_scale  # at this point the split_point has no effect
		max_tput_dset = original_max_tput_dset
		cfg_q, query_throughput = load_qtput_per_kscell(exp_config)  # (tr, cGridCell)
		
		a = int(actions[-1])
		"""By placing the [ith] grid cell, at the [a]th place in the machine
		you take the grid feature of the ith cell currently,
		i am just replacing the values, what happens if they have diff value or whether this is not possible at all
		actions.append(a)
		print(current_rtg)"""
		
		"""How does the state update?"""
		numa_machine_obs[int(a)] = True
		"""If you want the view mask to be a counter rather than a binary matrix"""
		# numa_machine_obs[int(a)] += 1
		"""If you want the position mask to be a counter rather than a binary matrix"""
		obs_mask_core[int(a)] -= 1

		if not(exp_config.num_meta_features == 0):
				o1, o2, o3 = get_state_up(actions[-1], len(actions)-1, exp_config)
		else:
				o1, o2 = get_state(actions[-1], len(actions)-1, exp_config)
		
		if o1.shape[0] == 0:
				print("Not Located!")
				numa_machine_obs_s[:, int(a)] += copy.deepcopy(seen_obss_s[:, seen_a_])
				tg_return[0] = current_rtg[-1] - st_return[len(actions)-1]
		else:
				print("located")
				numa_machine_obs_s[:, int(a)] += o1
				tg_return[0] = current_rtg[-1] - o2

				 
		numa_machine_obs = numa_machine_obs.view(-1, chassis_dimx, chassis_dimy)
		numa_machine_obs_s = numa_machine_obs_s.view(-1, chassis_dimx, chassis_dimy)
		
		"""If you want the position mask to be a counter rather than a binary matrix
				and balance the load
		"""
		curr_ts = len(actions)
		# print("TS = ", curr_ts)
		# if curr_ts == exp_config.cnt_grid_cells/2:
		#     bound_core = int(exp_config.cnt_grid_cells / exp_config.machine.num_worker)+1
		#     cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
		#     obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
		#     chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
		#     obs_mask_core[np.array(chassis_act_).astype(int)] = bound_core
				
		state_obs_mask = np.full((chassis_dimx * chassis_dimy,), False)
		mask_already_full = np.where(obs_mask_core == 0)
		state_obs_mask[mask_already_full] = True
		state_obs_mask = np.reshape(state_obs_mask, (1, chassis_dimx, chassis_dimy))
		state_obs_mask = torch.tensor(state_obs_mask)
		numa_machine_obss_mask = state_obs_mask.view(-1, chassis_dimx, chassis_dimy)
		numa_machine_obss_mask = numa_machine_obss_mask.view(-1, chassis_dimx, chassis_dimy)
		obs_state_new = torch.cat((numa_machine_obs, numa_machine_obs_s, numa_machine_obss_mask), dim=0).unsqueeze(0)
		
		# print(obs_state_new.shape)

		obs_state = torch.cat((current_x, obs_state_new), dim=0)
		# print(state.shape)

		rtgs = torch.cat((current_rtg, tg_return))
		
		rtgs = rtgs / exp_config.rtg_div
		
		# print(reward.shape)
		
		if not(exp_config.num_meta_features == 0):
				if o1.shape[0] == 0:
						print(m_x.shape, current_mx.shape)
						mx = m_x.unsqueeze(0)
						print(m_x.shape, current_mx.shape)
						metas = torch.cat((current_mx, mx), dim=0)
				else:
						mx = o3.unsqueeze(0)
						p_feat = [-1, -1]
						for key in processor_dict:
								if key in exp_config.processor:
									p_feat[0] = processor_dict[key]    
									break
						p_feat[1] = exp_config.machine.numa_node
						grid_features_p = torch.tensor([p_feat], dtype=mx.dtype, device=mx.device)
						
						mx = torch.cat([mx, grid_features_p], dim=1)
						print(mx.shape, current_mx.shape)
						metas = torch.cat((current_mx, mx), dim=0)
		else:
				mx = current_mx[-1].unsqueeze(0)
				print(mx.shape, current_mx.shape)
				metas = torch.cat((current_mx, mx), dim=0)


		# print(metas.shape)

		done = False 
		return obs_state, rtgs, done, metas, obs_mask_core


