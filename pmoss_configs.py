import os 
from enum import Enum

class Machine:
  def __init__(self, name):
    self.name = name
    file_hw = os.path.join("/home/yrayhan/works/lpmoss/machines/" + self.name + "/hw.cfg")    
    hw_cfgs = []
    
    with open(file_hw, 'r') as f:
      for line in f:
        line = line.strip()
        int_array = [int(_) for _ in line.split()]
        hw_cfgs.append(int_array)
    
    self.numa_node = hw_cfgs[0][0]
    self.worker_per_numa = hw_cfgs[0][1]
    self.mc_channel_per_numa = hw_cfgs[0][2]
    self.socket = hw_cfgs[0][3]
    self.num_upi_per_socket = hw_cfgs[0][4]
    self.num_worker = hw_cfgs[1][0] 
    self.li_ncore_dumper = hw_cfgs[2]
    self.li_worker = hw_cfgs[3]
    self.worker_to_chassis_pos_mapping  = hw_cfgs[4]
    assert(len(self.li_worker) == len(self.worker_to_chassis_pos_mapping))
  
  def __repr__(self):
        return (f"Machine(name={self.name}, cnt_numa_node={self.numa_node}, worker_per_numa={self.worker_per_numa}, "
                f"mc_channel_per_numa={self.mc_channel_per_numa}, num_worker={self.num_worker}, "
                f"li_ncore_dumper={self.li_ncore_dumper}, li_worker={self.li_worker}, "
                f"chassis mapping = {self.worker_to_chassis_pos_mapping})")


class ExpConfig:
    def __init__(self, processor, chassis_dim, index, workload, num_features, num_meta_features,
                 cnt_grid_cells, cfg_par, per_cfg_sample, policy_dim, 
                 rtg_scale, rtg_div, eval_start_cfg, idx_kb_folder):
        self.processor = processor
        self.chassis_dim = chassis_dim  # A tuple (x, y)
        self.index = index  # The name of the index used in the experiment
        self.workload = workload  # The type of workload used (e.g., read-heavy, write-heavy)
        self.num_features = num_features  # Number of features in the experiment configuration
        self.num_meta_features = num_meta_features
        self.cnt_grid_cells = cnt_grid_cells  # Number of grid cells for partitioning or sampling
        self.cfg_par = cfg_par # From which index does the data starts (including)
        self.per_cfg_sample = per_cfg_sample
        self.policy_dim = policy_dim
        self.rtg_scale = rtg_scale  # return to go 1.0, 1.1
        self.rtg_div = rtg_div  # Division by ?
        self.eval_start_cfg = eval_start_cfg  # For evaluation, which config to base it upon
        self.idx_kb_folder = idx_kb_folder  # kb_b, kb_r

        self.machine = Machine(self.processor)
        vendor, cpu = self.processor.split('_', 1)
        self.kb_path = os.path.join("/home/yrayhan/works/lpmoss/kbs/" + 
                                               vendor + "/" + cpu + "/" + 
                                               self.idx_kb_folder +"/"
                                               )    
        


    def __repr__(self):
      return (f"ExpConfig(processor={self.processor}\n"
              f"chassis_dim={self.chassis_dim}\n" 
              f"index={self.index}\n"
              f"workload={self.workload}\n"
              f"num_features={self.num_features}\n"
              f"num_meta_features={self.num_meta_features}\n"
              f"cnt_grid_cells={self.cnt_grid_cells}\n" 
              f"cfg_par={self.cfg_par}\n" 
              f"rtg_scale={self.rtg_scale}, rtg_div={self.rtg_div}\n"
              f"eval_start_cfg={self.eval_start_cfg}\n"
              f"idx_kb_folder={self.idx_kb_folder}\n"
              f"machine={self.machine}\n"
              f"kb_path={self.kb_path}")




class wl(Enum):
    MD_RS_UNIFORM = 0
    MD_RS_NORMAL = 1
    MD_LK_UNIFORM = 2
    MD_RS_ZIPF = 3
    MD_RS_HOT3 = 4
    MD_RS_HOT5 = 5
    MD_RS_HOT7 = 6
    MD_LK_RS_25_75 = 7
    MD_LK_RS_50_50 = 8
    MD_LK_RS_75_25 = 9
    MD_RS_LOGNORMAL = 10
    SD_YCSB_WKLOADA = 11
    SD_YCSB_WKLOADC = 12
    SD_YCSB_WKLOADE = 13
    SD_YCSB_WKLOADF = 14
    SD_YCSB_WKLOADE1 = 15
    SD_YCSB_WKLOADH = 16
    SD_YCSB_WKLOADI = 17
    WIKI_WKLOADA = 18
    WIKI_WKLOADC = 19
    WIKI_WKLOADE = 20
    WIKI_WKLOADF = 21
    WIKI_WKLOADI = 22
    WIKI_WKLOADH = 23
    WIKI_WKLOADA1 = 24
    WIKI_WKLOADA2 = 25
    WIKI_WKLOADA3 = 26
    OSM_WKLOADA = 27
    OSM_WKLOADC = 28
    OSM_WKLOADE = 29
    OSM_WKLOADH = 30
    SD_YCSB_WKLOADA1 = 31
    SD_YCSB_WKLOADH11 = 32
    OSM_WKLOADA0 = 33
    SD_YCSB_WKLOADH1 = 34
    SD_YCSB_WKLOADH2 = 35
    SD_YCSB_WKLOADH3 = 36
    SD_YCSB_WKLOADH4 = 37
    SD_YCSB_WKLOADH5 = 38
    SD_YCSB_WKLOADA00 = 39
    SD_YCSB_WKLOADA01 = 40
    SD_YCSB_WKLOADC1 = 41

# Second Enum: Dataset Sources (starting from 0)
class ds(Enum):
    OSM_USNE = 0
    GEOLITE = 1
    BERLINMOD02 = 2
    YCSB = 3
    WIKI = 4
    FB = 5
    OSM_CELLIDS = 6

# Third Enum: Tree Types (starting from 0)
class db_index(Enum):
    BTREE = 0
    RTREE = 1
    QUADTREE = 2

    

