import sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import copy


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
    
    
    grid_features_idx = np.arange(0, exp_config.cnt_grid_cells)
    grid_features_idx = np.reshape(grid_features_idx, (1, grid_features_idx.shape[0]))
    grid_features_idx = np.repeat(grid_features_idx, grid_features.shape[0], axis=0)
    grid_features_idx = np.reshape(grid_features_idx, (-1, exp_config.cnt_grid_cells, 1))

    # Turn it into a categorical variable
    # gIdx = torch.from_numpy(grid_features_idx)
    # grid_features_idx_one_hot = torch.zeros(gIdx.shape[0], 100)
    # grid_features_idx_one_hot.scatter_(1, gIdx.unsqueeze(1), 1.0)
    # grid_features_idx_one_hot = grid_features_idx_one_hot.view(1, grid_features_idx_one_hot.shape[0], grid_features_idx_one_hot.shape[1])
    # grid_features_idx_one_hot = np.repeat(grid_features_idx_one_hot, grid_features.shape[0], axis=0)
    # grid_features_idx_one_hot = np.reshape(grid_features_idx_one_hot, (grid_features.shape[0], exp_config.cnt_grid_cells, -1))

    # t = torch.tensor(grid_features_idx, dtype=torch.int64)
    # grid_features_idx_one_hot = F.one_hot(x=t, num_classes=exp_config.cnt_grid_cells)
    # grid_features_idx_one_hot = grid_features_idx_one_hot.view(1, grid_features_idx_one_hot.shape[0],
    #                                                            grid_features_idx_one_hot.shape[1])
    # grid_features_idx_one_hot = np.repeat(grid_features_idx_one_hot, grid_features.shape[0], axis=0)
    # grid_features_idx_one_hot = np.reshape(grid_features_idx_one_hot, (grid_features.shape[0], exp_config.cnt_grid_cells, -1))
    # for _ in range(100):
    #     print(grid_features_idx_one_hot[0][_])
    
    grid_features = np.concatenate([grid_features, grid_features_idx], axis=2)
    # DF = pd.DataFrame(np.reshape(grid_features, (grid_features.shape[0], -1))) 
    # DF.to_csv("data1.csv")
    # grid_features = np.concatenate([grid_features, grid_features_idx_one_hot], axis=2)
    return cfgs_info, grid_features


def load_uncore_features_intel(exp_config):
    # Only for intel
    cfg_par = exp_config.cfg_par 
    num_numa = exp_config.machine.numa_node
    num_mc_per_numa = exp_config.machine.mc_channel_per_numa
    
    RAW_FILE = exp_config.kb_path + str(exp_config.machine.li_ncore_dumper[0]) + "/mem-channel_view.txt"
    raw_array = np.loadtxt(RAW_FILE)
    read_channels = raw_array[:, cfg_par:cfg_par+num_numa*num_mc_per_numa]
    write_channels = raw_array[:, cfg_par+num_numa*num_mc_per_numa:cfg_par+2*num_numa*num_mc_per_numa]
    
    cfg = raw_array[:, :cfg_par]

    read_channels = np.reshape(read_channels, (read_channels.shape[0], num_numa, -1))
    write_channels = np.reshape(write_channels, (write_channels.shape[0], num_numa, -1))

    read_channels_throughput_ts = np.sum(read_channels, axis=2)
    write_channels_throughput_ts = np.sum(write_channels, axis=2)

    read_channels_throughput_ts = np.reshape(read_channels_throughput_ts,
                                             (read_channels_throughput_ts.shape[0], num_numa, -1))
    write_channels_throughput_ts = np.reshape(write_channels_throughput_ts,
                                              (write_channels_throughput_ts.shape[0], num_numa, -1))

    return cfg, read_channels_throughput_ts, write_channels_throughput_ts


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


def find_correct_max_tput_for_wl(wl, exp_config):
    cfg_q2, query_throughput_numa = load_qtput_cum(exp_config)  # (tr, )
    valid_tputs = []
    valid_idxs = []
    valid_cfgs = []
    
    
    for _ in range(query_throughput_numa.shape[0]):  # tr
        cfg_ = cfg_q2[0][_][0]
        wl_ = cfg_q2[0][_][2]
        if wl_ == wl:
            valid_tputs.append(query_throughput_numa[_])
            valid_cfgs.append(cfg_)
            valid_idxs.append(_)
    
    tputs_to_consider = np.asarray(valid_tputs)
    max_tput = np.max(tputs_to_consider)
    mIdx = np.argmax(tputs_to_consider)
    
    maxCfg = valid_cfgs[mIdx]
    maxIdx = valid_idxs[mIdx]    
    
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


def load_actions(exp_config, cfg, onlyNUMA=False):
    
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
    if exp_config.cnt_grid_cells == 100:
        machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + ".txt"
    else:
        machine_config_path = "./machine_configs/" + exp_config.processor + "/" + "c_" + str(cIdx) + "_" + str(exp_config.cnt_grid_cells) + ".txt"
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


def retrieve_config(cfg, idx):
    if machine == 0:
        cntNUMANode = 8
        cntCorePerNUMA = 8
        nGridCell = 10
    if machine == 1:
        cntNUMANode = 2
        cntCorePerNUMA = 40
        nGridCell = 10

    thread_range_indexed = [
        _ for _ in range(0, cntNUMANode*cntCorePerNUMA)
    ]
    if machine == 0:
        thread_range_original = [
            [_ for _ in range(5, 13)],
            [_ for _ in range(17, 25)],
            [_ for _ in range(29, 37)],
            [_ for _ in range(41, 49)],
            [_ for _ in range(53, 61)],
            [_ for _ in range(65, 73)],
            [_ for _ in range(77, 85)],
            [_ for _ in range(89, 97)]
        ]
    else:
        thread_range_original = [
            [_ for _ in range(6, 86, 2)],
            [_ for _ in range(7, 87, 2)]
        ]

    gen_core_dict = {}
    gen_numa_dict = {}
    coreIdx = 0
    if machine == 0:    
        for _ in range(len(thread_range_indexed)):
            idx_core = int(thread_range_indexed[_])
            numa_node = (int) (idx_core / cntNUMANode)
            numa_core = (int) (idx_core % cntCorePerNUMA)
            gen_core_dict[idx_core] = thread_range_original[numa_node][numa_core]
            gen_numa_dict[idx_core] = numa_node
    else:
        t = 0
        for _ in range(len(thread_range_indexed)):
            if _ < 40: 
                gen_numa_dict[_] = 0
                gen_core_dict[_] = _*2 + 6
            
            else:
                
                gen_numa_dict[_] = 1
                gen_core_dict[_] = t*2 + 6 + 1
                t += 1
    
    # config_numa = np.reshape(config_numa, (config_numa.shape[0] * config_numa.shape[1],))
    # config_core = np.reshape(nGridCell, (config_core.shape[0] * config_core.shape[1],))
    if machine == 0:
        retrieved_configs_core = [_ for _ in range(0, nGridCell*nGridCell)]
        for _ in range(len(cfg)):
            retrieved_configs_core[_] = gen_core_dict[cfg[_]]

        retrieved_configs_numa = [_ for _ in range(0, nGridCell*nGridCell)]
        for _ in range(len(cfg)):
            retrieved_configs_numa[_] = gen_numa_dict[cfg[_]]
        
        retrieved_configs_core = np.asarray(retrieved_configs_core)
        retrieved_configs_core = np.reshape(retrieved_configs_core, (nGridCell, nGridCell))

        retrieved_configs_numa = np.asarray(retrieved_configs_numa)
        retrieved_configs_numa = np.reshape(retrieved_configs_numa, (nGridCell, nGridCell))

        retrieved_configs = np.vstack((retrieved_configs_numa, retrieved_configs_core))
        retrieved_configs = retrieved_configs.astype(int)
        
        np.savetxt("l-machine-configs/config_" + str(idx) + ".txt", retrieved_configs, fmt='%i')
    else:
        retrieved_configs_core = [_ for _ in range(0, nGridCell*nGridCell)]
        for _ in range(len(cfg)):
            retrieved_configs_core[_] = gen_core_dict[cfg[_]]

        retrieved_configs_numa = [_ for _ in range(0, nGridCell*nGridCell)]
        for _ in range(len(cfg)):
            retrieved_configs_numa[_] = gen_numa_dict[cfg[_]]
        
        retrieved_configs_core = np.asarray(retrieved_configs_core)
        retrieved_configs_core = np.reshape(retrieved_configs_core, (nGridCell*nGridCell))

        retrieved_configs_numa = np.asarray(retrieved_configs_numa)
        retrieved_configs_numa = np.reshape(retrieved_configs_numa, (nGridCell*nGridCell))

        retrieved_configs = np.vstack((retrieved_configs_numa, retrieved_configs_core))
        retrieved_configs = retrieved_configs.astype(int)
        
        retrieved_configs = np.reshape(retrieved_configs, (nGridCell*2, nGridCell))
        np.savetxt("l-machine-configs/config_" + str(idx) + ".txt", retrieved_configs, fmt='%i')
    return retrieved_configs

# osm, quadtree, uni-scan, 4S8N
# li1 = [58, 52, 18, 5, 56, 16, 9, 42, 30, 34, 40, 46, 0, 37, 60, 51, 2, 22, 25, 26, 43, 50, 62, 57, 6, 3, 21, 31, 14, 55, 38, 24, 49, 8, 29, 13, 53, 39, 15, 59, 1, 61, 47, 35, 20, 41, 32, 44, 4, 10, 54, 36, 48, 11, 28, 12, 23, 7, 63, 17, 19, 33, 27, 45, 39, 15, 39, 20, 32, 29, 5, 24, 38, 46, 21, 5, 51, 38, 37, 12, 27, 49, 12, 19, 21, 0, 12, 49, 22, 56, 46, 14, 63, 30, 56, 26, 56, 29, 44, 6]
# retrieve_config(li1, 300010)  # 40
# li1 = [0, 6, 8, 16, 12, 2, 13, 20, 30, 14, 28, 33, 41, 17, 37, 43, 50, 62, 4, 53, 58, 10, 21, 51, 56, 44, 45, 46, 27, 5, 11, 9, 52, 42, 61, 24, 40, 22, 54, 48, 19, 38, 29, 35, 26, 3, 49, 59, 57, 34, 39, 23, 1, 47, 55, 32, 18, 25, 15, 31, 36, 60, 7, 63, 51, 5, 12, 24, 49, 45, 22, 19, 22, 33, 40, 20, 13, 20, 46, 12, 27, 50, 42, 26, 21, 0, 19, 9, 15, 6, 7, 62, 63, 30, 56, 32, 59, 49, 44, 6]
# retrieve_config(li1, 300020)  # 30
# # glife, quadtree, uni-scan, 4S8N
# li3 = [0, 6, 8, 16, 12, 2, 13, 20, 30, 14, 28, 33, 41, 17, 50, 62, 51, 56, 5, 24, 44, 43, 48, 58, 49, 27, 21, 52, 53, 57, 34, 61, 18, 59, 54, 36, 60, 35, 45, 42, 46, 19, 29, 1, 40, 4, 38, 3, 9, 63, 32, 23, 25, 11, 22, 26, 37, 39, 15, 31, 55, 47, 7, 10, 51, 32, 29, 24, 49, 45, 17, 19, 22, 24, 44, 61, 13, 20, 46, 12, 27, 50, 42, 19, 21, 0, 19, 9, 15, 6, 7, 62, 63, 30, 46, 25, 59, 29, 44, 6]
# retrieve_config(li3, 300011)  # 40
# bmod02, quadtree, uni-scan, 4S8N
# li2 = [58, 52, 18, 5, 56, 16, 9, 42, 30, 34, 40, 46, 0, 37, 60, 51, 2, 22, 25, 26, 43, 50, 62, 57, 6, 3, 21, 31, 14, 55, 38, 24, 49, 27, 54, 61, 35, 45, 1, 59, 53, 39, 47, 28, 41, 32, 29, 44, 7, 15, 20, 36, 8, 63, 4, 13, 12, 48, 23, 10, 19, 11, 17, 33, 39, 39, 39, 39, 55, 15, 39, 55, 39, 55, 55, 15, 22, 15, 15, 20, 15, 32, 5, 46, 5, 38, 37, 9, 56, 6, 46, 62, 3, 50, 46, 56, 42, 27, 6, 42]
# retrieve_config(li2, 300020)  # 40
# li2 = [58, 52, 18, 5, 56, 16, 9, 42, 30, 34, 40, 46, 0, 37, 60, 51, 2, 22, 25, 26, 43, 50, 54, 62, 57, 6, 3, 21, 14, 31, 35, 32, 38, 27, 12, 49, 61, 24, 28, 44, 8, 41, 29, 45, 23, 55, 39, 15, 47, 7, 20, 36, 4, 13, 48, 53, 59, 63, 1, 10, 11, 33, 17, 19, 0, 4, 39, 39, 49, 12, 39, 47, 39, 47, 55, 55, 39, 39, 55, 55, 55, 55, 32, 19, 21, 29, 24, 22, 32, 5, 0, 26, 38, 30, 49, 42, 56, 12, 19, 6]
# [58, 52, 18, 5, 56, 16, 9, 42, 30, 34, 40, 46, 0, 37, 60, 51, 2, 22, 25, 26, 43, 50, 54, 62, 57, 6, 3, 21, 14, 31, 35, 32, 38, 27, 12, 49, 61, 24, 28, 44, 8, 41, 29, 45, 23, 55, 39, 15, 47, 7, 20, 36, 4, 13, 48, 53, 59, 63, 1, 10, 11, 33, 17, 19, 0, 4, 39, 39, 49, 12, 39, 47, 39, 47, 55, 55, 39, 39, 55, 55, 55, 55, 32, 19, 21, 29, 24, 22, 32, 5, 0, 26, 38, 30, 49, 42, 56, 12, 19, 6]
# retrieve_config(li2, 300022)  # 30
# li_ = [31, 48, 4, 42, 57, 43, 5, 6, 0, 1, 2, 3, 44, 46, 45, 29, 34, 61, 7, 23, 40, 15, 41, 70, 49, 33, 39, 27, 50, 11, 13, 60, 24, 35, 25, 59, 78, 32, 14, 22, 36, 18, 53, 52, 12, 76, 64, 38, 10, 47, 55, 56, 58, 37, 68, 73, 65, 66, 28, 21, 72, 8, 77, 51, 62, 26, 75, 54, 69, 9, 67, 19, 71, 20, 16, 63, 17, 74, 30, 79, 12, 29, 38, 7, 1, 45, 69, 29, 39, 54, 24, 22, 18, 33, 17, 7, 23, 42, 18, 47]
# li_ = [31, 48, 4, 42, 57, 43, 5, 6, 0, 1, 2, 3, 44, 45, 46, 40, 41, 61, 64, 47, 8, 58, 79, 70, 28, 11, 23, 27, 14, 7, 13, 60, 29, 35, 49, 15, 20, 39, 25, 21, 12, 69, 59, 33, 56, 55, 34, 10, 71, 53, 73, 63, 18, 62, 67, 36, 65, 9, 72, 19, 51, 16, 26, 38, 76, 22, 68, 54, 32, 66, 50, 37, 17, 78, 52, 77, 24, 74, 30, 75, 12, 29, 38, 7, 1, 45, 5, 29, 39, 54, 24, 22, 18, 33, 17, 7, 23, 42, 18, 47]
# retrieve_config(li_, 10000000)  # 30
# li = [33, 56, 54, 18, 5, 59, 17, 48, 4, 21, 40, 53, 42, 30, 9, 34, 57, 46, 41, 22, 27, 52, 25, 37, 26, 44, 51, 62, 43, 35, 6, 49, 60, 3, 10, 2, 36, 8, 32, 58, 1, 19, 38, 24, 61, 12, 0, 45, 20, 50, 28, 29, 47, 16, 63, 11, 23, 39, 14, 13, 15, 55, 31, 7, 27, 35, 6, 63, 30, 9, 44, 35, 6, 9, 34, 61, 44, 6, 46, 25, 34, 59, 44, 20, 44, 6, 19, 9, 15, 6, 7, 62, 30, 14, 9, 9, 44, 44, 44, 6]
# lice = [31, 48, 4, 42, 57, 43, 5, 6, 0, 1, 2, 3, 44, 45, 46, 40, 41, 61, 64, 47, 8, 58, 79, 70, 28, 11, 23, 27, 14, 7, 13, 60, 29, 35, 49, 20, 39, 25, 21, 12, 69, 59, 33, 56, 55, 34, 10, 71, 53, 63, 18, 62, 67, 36, 65, 9, 72, 19, 51, 16, 26, 38, 76, 22, 68, 54, 32, 66, 37, 17, 78, 24, 73, 15, 52, 77, 50, 74, 30, 75, 12, 29, 38, 7, 1, 45, 5, 29, 39, 54, 24, 22, 18, 33, 17, 7, 23, 42, 18, 47]
# retrieve_config(lice, 9990)  # 40
# li = [47, 28, 45, 53, 17, 11, 60, 52, 1, 2, 44, 18, 58, 59, 3, 12, 10, 33, 19, 6, 30, 43, 54, 0, 9, 36, 42, 37, 40, 38, 27, 25, 56, 48, 61, 35, 62, 4, 13, 41, 24, 32, 46, 49, 16, 22, 50, 5, 14, 20, 51, 26, 57, 21, 8, 29, 34, 39, 23, 63, 15, 7, 31, 55, 31, 15, 15, 31, 63, 31, 15, 15, 15, 55, 55, 63, 51, 51, 15, 31, 31, 15, 31, 55, 15, 31, 31, 15, 15, 31, 7, 31, 31, 31, 31, 25, 51, 32, 44, 6]
# retrieve_config(li, 500000)  # 41
# li = [47, 28, 45, 53, 17, 11, 60, 52, 1, 2, 44, 18, 58, 59, 3, 12, 10, 33, 19, 6, 30, 43, 54, 0, 9, 36, 42, 37, 40, 38, 27, 25, 56, 48, 61, 35, 62, 4, 13, 41, 24, 32, 46, 49, 16, 22, 50, 5, 14, 20, 51, 26, 57, 21, 8, 29, 34, 39, 23, 15, 63, 55, 7, 31, 63, 15, 15, 31, 63, 63, 15, 51, 31, 55, 55, 55, 15, 31, 15, 55, 55, 15, 31, 51, 32, 0, 19, 9, 15, 6, 24, 62, 63, 30, 46, 25, 59, 29, 44, 6]
li = [57, 2, 3, 0, 17, 9, 8, 5, 10, 18, 24, 12, 28, 32, 42, 21, 26, 35, 46, 52, 62, 6, 53, 44, 36, 49, 58, 56, 60, 4, 13, 19, 45, 34, 61, 22, 40, 41, 48, 11, 25, 43, 14, 33, 20, 47, 29, 55, 16, 38, 27, 37, 1, 59, 50, 51, 54, 30, 15, 63, 23, 7, 31, 39, 34, 4, 17, 40, 48, 58, 45, 56, 60, 49, 34, 41, 43, 59, 33, 22, 27, 38, 1, 12, 45, 52, 53, 37, 13, 20, 18, 25, 43, 50, 35, 57, 38, 29, 41, 21]
li = [47, 0, 2, 13, 21, 5, 12, 4, 14, 28, 30, 34, 42, 29, 6, 11, 10, 22, 24, 32, 33, 40, 53, 44, 51, 61, 56, 18, 43, 50, 60, 3, 58, 1, 9, 19, 20, 35, 45, 49, 36, 37, 17, 26, 25, 41, 54, 59, 38, 46, 27, 52, 62, 57, 8, 48, 16, 39, 15, 55, 63, 23, 31, 7, 51, 28, 29, 14, 60, 45, 17, 19, 22, 44, 44, 61, 13, 20, 46, 58, 27, 9, 47, 47, 21, 49, 19, 9, 15, 35, 7, 38, 38, 36, 46, 25, 59, 29, 44, 6]
li = [3, 6, 14, 19, 26, 33, 44, 49, 62, 57, 13, 4, 18, 25, 32, 46, 53, 56, 34, 43, 54, 36, 0, 27, 51, 58, 41, 10, 21, 28, 29, 1, 48, 61, 2, 38, 12, 59, 40, 42, 50, 60, 5, 30, 16, 37, 45, 52, 17, 20, 24, 63, 8, 22, 9, 35, 15, 11, 23, 31, 55, 47, 7, 39, 63, 34, 34, 0, 36, 46, 57, 63, 25, 60, 33, 5, 13, 20, 26, 27, 43, 36, 53, 48, 57, 4, 13, 22, 27, 38, 35, 27, 45, 52, 56, 53, 10, 16, 26, 38]
li = [3, 6, 14, 19, 26, 33, 44, 49, 62, 57, 13, 4, 18, 25, 32, 46, 53, 56, 34, 43, 54, 36, 51, 58, 27, 38, 1, 11, 28, 5, 12, 48, 40, 20, 59, 41, 60, 0, 10, 22, 29, 30, 45, 63, 37, 61, 24, 42, 35, 50, 21, 17, 8, 55, 39, 7, 15, 47, 31, 52, 9, 23, 16, 2, 51, 32, 0, 11, 54, 33, 28, 14, 57, 58, 12, 27, 38, 18, 34, 27, 43, 35, 46, 61, 6, 24, 61, 44, 49, 36, 25, 6, 10, 13, 60, 20, 8, 29, 30, 41]
# retrieve_config(li, -7)  # 41

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
        cfg_q3, read_channels_throughput_ts, write_channels_throughput_ts = load_uncore_features_intel(exp_config)
        mc_tput = np.concatenate([read_channels_throughput_ts, write_channels_throughput_ts], axis=2)
    
    grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
    scaler = StandardScaler()
    grid_features = scaler.fit_transform(grid_features)
    grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features+1))

    if not(exp_config.num_meta_features == 0):
        mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], -1))
        scaler_mc = StandardScaler()
        mc_tput = scaler_mc.fit_transform(mc_tput)
        mc_tput = np.reshape(mc_tput, (mc_tput.shape[0], 1, -1))
        mc_tput = np.repeat(mc_tput, exp_config.cnt_grid_cells, axis=1)
    
    orginal_max_tput_dset = find_correct_max_tput(exp_config) * exp_config.rtg_scale
    # orginal_max_tput_dset = find_correct_max_tput_for_wl(exp_config) * exp_config.rtg_scale

    obss = []
    obss_s = []
    obss_mask = []
    actions = []
    stepwise_returns = []
    rtgs = []
    done_idxs = []
    timesteps = []
    metas = []
    
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
        act_ = load_actions(exp_config, cfg_)
        actions.append(act_)

        if not(exp_config.num_meta_features == 0):
            # Load the meta mc_data
            metas.append(mc_tput[_])

        # TODO: This needs some serious thought
        # obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 3)

        numa_machine_obs = np.full((chassis_dimx * chassis_dimy, ), False)
        numa_machine_obs_s = np.full((chassis_dimx * chassis_dimy, exp_config.num_features+1), 0, dtype=np.float64)
        # numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), True)
        
        numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
        obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
        chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
        obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
        mask_already_full = obs_mask_core.nonzero()
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
        for i in range(act_.shape[0]):  # each timestep 
            # map_idx=exp_config.machine.li_worker.index(int(act_[i]))
            # a = exp_config.machine.worker_to_chassis_pos_mapping[map_idx]
            a = int(cores_position[int(act_[i])])
        
            # => Add this if condition so that i can place stuff again
            # if i > 32:
            #     obs_mask_core = np.full((num_numa * num_worker_per_numa, ), 2)
            # if i > 60 and not(flag):  # For obs_mask_core = [2+1]
            #     obs_mask_core = np.full((num_numa * num_worker_per_numa, ), 1)
            #     flag = True

            # numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
            # mask_already_full = obs_mask_core.nonzero()
            # numa_machine_obss_mask[mask_already_full] = True

            numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
            mask_already_full = obs_mask_core.nonzero()
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
    
    return obss, obss_s, obss_mask, actions, stepwise_returns, rtgs, done_idxs, timesteps, metas, lengths, benchmarks



def gen_token(exp_config):
    
    # (tr, 2) (tr, nGridCells, nFeatures) (tr, 100, 5)
    idx_array, grid_features = load_hardware_snapshot(exp_config) # hardware snapshot + grid index 
    cfg_q, query_throughput = load_qtput_per_kscell(exp_config) # (tr, cGridCell)
    cfg_q2, query_throughput_numa = load_qtput_cum(exp_config) # (tr, )
    if not(exp_config.num_meta_features == 0):
        cfg_q3, read_channels_throughput_ts, write_channels_throughput_ts = load_uncore_features_intel(exp_config)
        mc_tput = np.concatenate([read_channels_throughput_ts, write_channels_throughput_ts], axis=2)
    
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
    scaler = StandardScaler()
    grid_features = scaler.fit_transform(grid_features)
    grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features+1))

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
    
    num_numa = exp_config.machine.numa_node
    num_worker_per_numa = exp_config.machine.worker_per_numa
    lbound_core = 2 # 100 / 56
    lbound_numa = 13 # 100/8
    chassis_dimx = exp_config.chassis_dim[0]
    chassis_dimy = exp_config.chassis_dim[1]

    # TODO: Return timesteps, resolve the last state thingy
    # _ = each full sample: we are breaking it into timesteps
    
    cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
    
    for _ in range(idx_array.shape[0]):  # tr
        cfg_ = idx_array[_][0]
        # print(cfg_)
        # Load the actions (how many for each complete row? = no of grid cells)
        act_ = load_actions(exp_config, cfg_)
        actions.append(act_)

        if not(exp_config.num_meta_features == 0):
            # Load the meta mc_data
            metas.append(mc_tput[_])

        
        # TODO: 
        # obs_mask_core = np.full((num_numa * num_worker_per_numa, ), lbound_core)
        # obs_mask_numa = np.full((num_numa,), lbound_numa)
        
        

        numa_machine_obs = np.full((chassis_dimx * chassis_dimy, ), False)
        numa_machine_obs_s = np.full((chassis_dimx * chassis_dimy, exp_config.num_features+1), 0, dtype=np.float64)
        numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), True)
        obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 3)
        # obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
        # obs_mask_core[np.array(act_).astype(int)] = 1
        
        # numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
        # obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
        # chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
        # obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
        # mask_already_full = obs_mask_core.nonzero()
        # numa_machine_obss_mask[mask_already_full] = True
        

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
        for i in range(act_.shape[0]):  # each timestep 
            # since we already refine it to 0 - 63, we do not need it
            # map_idx=exp_config.machine.li_worker.index(int(act_[i]))
            # a = exp_config.machine.worker_to_chassis_pos_mapping[map_idx]
            
            a = int(cores_position[int(act_[i])])
                
            
            # => Add this if condition so that i can place stuff again
            # if i > 32:
            #     obs_mask_core = np.full((num_numa * num_worker_per_numa, ), 2)
            
            # if i > 50 and not(flag):  # For obs_mask_core = [2+1]
            #     obs_mask_core = np.full((chassis_dimx * chassis_dimy, ), 0)
            #     chassis_act_=[int(cores_position[int(z)]) for z in range(exp_config.machine.num_worker)]
            #     obs_mask_core[np.array(chassis_act_).astype(int)] = lbound_core
            #     flag = True

                        
            numa_machine_obss_mask = np.full((chassis_dimx * chassis_dimy,), False)
            mask_already_full = obs_mask_core.nonzero()
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
    
    actions = np.reshape(np.asarray(actions), (-1, ))  # (nsamples * ngridcells, 1)
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
    
    return obss, obss_s, obss_mask, actions, stepwise_returns, rtgs, done_idxs, timesteps, metas, lengths, benchmarks





def get_state(action, ts, exp_config):
    idx_array, grid_features = load_hardware_snapshot(exp_config)
    cfg_q, query_throughput = load_qtput_per_kscell(exp_config)  # (tr, cGridCell)
    
    
    # ranges = get_wkload_range('ycsb', 3, 6, '4S8N')
    # ranges  = [(ranges[0][0], query_throughput.shape[0])]
    # print(ranges)
    # idx_array_tem = idx_array[ranges[0][0]-1: ranges[0][1], :]
    # grid_features_tem = grid_features[ranges[0][0]-1: ranges[0][1], :, :]
    # query_throughput_tem = query_throughput[ranges[0][0]-1: ranges[0][1], :]
    # print(idx_array_tem.shape)
    # for _ in range(1, len(ranges)):
    #     idx_array_tem = np.concatenate([idx_array_tem, idx_array[ranges[_][0]-1: ranges[_][1], :]], axis=0)
    #     grid_features_tem = np.concatenate([grid_features_tem, grid_features[ranges[_][0]-1: ranges[_][1], :]], axis=0)
    #     query_throughput_tem = np.concatenate([query_throughput_tem, query_throughput[ranges[_][0]-1: ranges[_][1], :]], axis=0)
    # (2492, 4) (2492, 100, 16) (2492, 100)
    # idx_array = idx_array_tem
    # grid_features = grid_features_tem
    # query_throughput = query_throughput_tem
    
    refined_idx = np.where(idx_array[:, 2] == exp_config.workload)[0]
    idx_array = idx_array[refined_idx]
    grid_features = grid_features[refined_idx]
    query_throughput = query_throughput[refined_idx]

    grid_features = np.reshape(grid_features, (grid_features.shape[0], -1))
    scaler = StandardScaler()
    grid_features = scaler.fit_transform(grid_features)
    grid_features = np.reshape(grid_features, (grid_features.shape[0], -1, exp_config.num_features+1))

    actions = []
    for _ in range(idx_array.shape[0]):
        a_ = load_actions(exp_config, idx_array[_][0])
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
    # ret_query_tput = torch.tensor(ret_query_tput)
    
    return ret_state, ret_query_tput


def env_update(
        x, m_x, st_return, actions, 
        current_x, current_mx, current_rtg, exp_config, 
        cfgIdx=41, board_formation=True, rtg_scale = 1.2
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
    
    
    
    seen_obs = x.view(-1, chassis_dimx, chassis_dimy)
    seen_obss_s = seen_obs[1:num_features+2, :, :].view(-1, chassis_dimx*chassis_dimy)
    
    # print(seen_obs.shape, seen_obss_s.shape, seen_st.shape)
    
    # seen_st = st  # st is un-normalized
    
    obss = []
    obss_s = []
    obss_mask = []
    stepwise_returns = []
    rtgs = []
    metas = []
    done_idxs = []
    timesteps = []
    
    cfg_ = exp_config.eval_start_cfg
    cores_position = exp_config.machine.worker_to_chassis_pos_mapping 
    
    # Load the actions (how many for each complete row? = no of grid cells)
    # This is the one you are currently seeing
    seen_act_ = load_actions(exp_config, cfg_)
    
    
    # Load the meta mc_data
    # metas.append(mc_tput[_])

    # TODO: This needs some serious thought
    # print(current_x.shape, current_mx.shape, current_rtg.shape)
    
    numa_machine_obs = current_x[-1][0, :, :].view(-1, )
    numa_machine_obs_s = current_x[-1][1:num_features+2, :, :].view(-1, chassis_dimx*chassis_dimy)
    numa_machine_obss_mask = current_x[-1][num_features+1, :, :].view(-1, )
    tg_return = torch.tensor([0])
    

    # print(numa_machine_obs.shape, numa_machine_obs_s.shape, numa_machine_obss_mask.shape)
    
    cfg_q2, query_throughput_numa = load_qtput_cum(exp_config)  # (tr, )
    original_max_tput_dset = find_correct_max_tput(exp_config) * exp_config.rtg_scale  # at this point the split_point has no effect
    max_tput_dset = original_max_tput_dset
    cfg_q, query_throughput = load_qtput_per_kscell(exp_config)  # (tr, cGridCell)
    
    # actions is what the model is predicting: MODEL
    # seen_act_ is what we are currently observing as cfg_to_start
    # if not board_formation:
    #     a = int(actions[-1])
    #     seen_a_= int(seen_act_[-1])
    # else:
    # we are assuming the model predicts the board position anyways
    a = int(actions[-1])
    # a = int(cores_position[int(actions[i])])
    # map_idx=exp_config.machine.li_worker.index(int(seen_act_[-1]))
    # seen_a_ = exp_config.machine.worker_to_chassis_pos_mapping[map_idx]
    seen_a_= int(cores_position[int(seen_act_[-1])])
    
    
        
        
    # By placing the [ith] grid cell, at the [a]th place in the machine
    # you take the grid feature of the ith cell currently,
    # i am just replacing the values, what happens if they have diff value or whether this is not possible at all
    # actions.append(a)
    # print(current_rtg)
    numa_machine_obs[int(a)] = True
    o1, o2 = get_state(actions[-1], len(actions)-1, exp_config)

    # => Added these for zero-shot learning on a different workload
    # o1, o2 = torch.tensor([]), torch.tensor([])

    if o1.shape[0] == 0:
        print("Not Located!")
        numa_machine_obs_s[:, int(a)] = copy.deepcopy(seen_obss_s[:, seen_a_])
        tg_return[0] = current_rtg[-1] - st_return[len(actions)-1]
    else:
        print("located")
        numa_machine_obs_s[:, int(a)] = o1
        tg_return[0] = current_rtg[-1] - o2

         
    numa_machine_obs = numa_machine_obs.view(-1, chassis_dimx, chassis_dimy)
    numa_machine_obs_s = numa_machine_obs_s.view(-1, chassis_dimx, chassis_dimy)
    numa_machine_obss_mask = numa_machine_obss_mask.view(-1, chassis_dimx, chassis_dimy)
    
    
    obs_state_new = torch.cat((numa_machine_obs, numa_machine_obs_s, numa_machine_obss_mask), dim=0).unsqueeze(0)
    
    # print(obs_state_new.shape)

    obs_state = torch.cat((current_x, obs_state_new), dim=0)
    # print(state.shape)

    rtgs = torch.cat((current_rtg, tg_return))
    
    rtgs = rtgs / exp_config.rtg_div
    
    # print(reward.shape)
    
    mx = current_mx[-1].unsqueeze(0)
    metas = torch.cat((current_mx, mx), dim=0)
    
    # print(metas.shape)



    done = False 
    return obs_state, rtgs, done, metas


