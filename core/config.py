from yacs.config import CfgNode as CN

def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'ZINC'
    # Additional num of worker for data loading
    cfg.num_workers = 12
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0 
    # Additional string add to logging 
    cfg.handtune = ''
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # version 
    cfg.version = 'final'
    # task, for simulation datasets
    cfg.task = -1
    # amp 
    cfg.amp = False

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    # Total graph mini-batch size
    cfg.train.batch_size = 100
    # Maximal number of epochs
    cfg.train.epochs = 100
    # Number of runs with random init 
    cfg.train.runs = 3
    # Base learning rate
    cfg.train.lr = 0.001
    # number of steps before reduce learning rate
    cfg.train.lr_patience = 50
    # learning rate decay factor
    cfg.train.lr_decay = 0.5
    # L2 regularization, weight decay
    cfg.train.wd = 0.
    # Dropout rate
    cfg.train.dropout = 0.
    
    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    # GNN type used, see core.model_utils.pyg_gnn_wrapper for all options
    cfg.model.arch_type = 'KCSetGNN'  #[''SubgraphGNN', 'KCSetGNN', 'PPGN']
    cfg.model.gnn_type = 'GINEConv' # change to list later
    cfg.model.bgnn_type = 'Sequential'  # bipartite gnn propagation type
    # Hidden size of the model
    cfg.model.hidden_size = 128
    # Number of gnn layers (doesn't include #MLPs)
    cfg.model.num_layers = 4
    cfg.model.num_inners = 2
    # Pooling type for generaating graph/subgraph embedding from node embeddings 
    cfg.model.pool = 'add'

    cfg.model.half_step = False # bipartite propagation type 

    # ------------------------------------------------------------------------ #
    # Subgraph options
    # ------------------------------------------------------------------------ #
    cfg.subgraph = CN()
    cfg.subgraph.type = 'kWL' # ['kWL', 'cluster', 'ego']
    cfg.subgraph.kmax = 3
    cfg.subgraph.kmin = 0
    cfg.subgraph.stack = True
    cfg.subgraph.num_components = 1
    cfg.subgraph.zero_init = True # whether init multiple-components sets with 0

    return cfg
    
import os 
import argparse
# Principle means that if an option is defined in a YACS config object, 
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining, 
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.

def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, 
                         help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg 
    cfg = cfg.clone()
    
    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line 
    cfg.merge_from_list(args.opts)
       
    return cfg

"""
    Global variable
"""
cfg = set_cfg(CN())