"""
Reinforcement Learning
"""
import os, shutil
import time
import torch
import numpy as np

from models.model import TSPXL
from configs.default_config import args
from utils.exp_utils import create_exp_dir
from utils.data_utils import RandomTSPGenerator

# Set seed, cuda, directory, etc.
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_id
    device = torch.device('cuda')
    print(f'Total {torch.cuda.device_count()} GPUs in USE : total')
    for i in range(torch.cuda.device_count()):
        print(f'<< GPU : {torch.cuda.get_device_name(i)}, id: {args.gpu_id} >>')
else:
    device = torch.device('cpu')

# Set logger
scripts_to_save = ['train.py', 'models']
log = create_exp_dir(args.exp_dir, scripts_to_save, args.debug)

# Load Dataset and iterator
if args.n_point >= 50:
    pass
else:
    data_loader = RandomTSPGenerator()
    
# Load hyperparameters
# Initialize criterion and optimizer
# Initialize model
model = TSPXL(
    d_model=args.d_model,
    d_ff=args.d_ff,
    n_head=args.n_head,
    n_layer=args.n_layer,
    n_class=args.n_point,
    bsz=args.bsz,
    deterministic=args.deterministic,
    criterion=criterion,
    optimizer=optimizer,
    dropout_rate=args.dropout_rate,
    internal_drop=args.internal_drop,
    clip_value=args.clip_value,
    pre_lnorm=args.pre_lnorm,
    clamp_len=args.clamp_len
)
def evaluate():
    pass
def train():
    pass

