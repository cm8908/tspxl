"""

"""
from argparse import ArgumentError
import os, shutil
import time
import torch
import numpy as np

from torch import nn, optim
from models.model import TSPXL
from configs.default_rl import args
from utils.exp_utils import create_exp_dir
from utils.data_utils import RandomTSPGenerator, TSPDataset

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
if args.rl:
    data_loader = RandomTSPGenerator(segm_len=args.segm_len)
else:
    if args.n_point >= 50:
        train_set = TSPDataset(n=args.n_point, mode='train', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        test_set = TSPDataset(n=args.n_point, mode='test', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        train_set, val_set = train_set[:-len(test_set)], train_set[-len(test_set):]
    else:
        train_set = TSPDataset(n=args.n_point, mode='train', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        val_set = TSPDataset(n=args.n_point, mode='val', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        test_set = TSPDataset(n=args.n_point, mode='test', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
    pass
    
# Load hyperparameters
if args.rl and args.loss_fn in ['nll', 'ce', 'mse']:
    raise ValueError(f'Criterion `{args.loss_fn}` is not a valid method for RL')
if not args.rl and args.loss_fn in ['reinforce']:
    raise ValueError(f'Criterion `{args.loss_fn}` is not a valid method for SL')

# Criterion
if args.rl:
    if args.loss_fn == 'reinforce':
        def criterion(L_train, L_base, p):
            return torch.mean((L_train - L_base) * p)
else:
    if args.loss_fn == 'nll':
        criterion = nn.NLLLoss
    elif args.loss_fn == 'ce':
        criterion = nn.CrossEntropyLoss
    elif args.loss_fn == 'mse':
        criterion = nn.MSELoss

# Optimizer
if args.optim == 'sgd':
    optimizer = optim.SGD
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop
elif args.optim == 'adam':
    optimizer = optim.Adam
else:
    raise ValueError("Proper optimizer should be provided")

# Build model
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
if args.rl:
    baseline = TSPXL(
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

if args.multi_gpu:
    model = nn.DataParallel(model).to(device)
    try: baseline = nn.DataParallel(baseline).to(device)
    except NameError: pass
else:
    model = model.to(device)
    try: baseline = baseline.to(device)
    except NameError: pass

def compute_tour_length(tour, x): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (N, B, 2) batch of tsp tour instances
             tour of size (B, N) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    x = x.permute(1,0,2).contiguous()  # (B, N, 2)
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    arange_vec = torch.arange(bsz, device=x.device)
    first_cities = x[arange_vec, tour[:,0], :] # size(first_cities)=(bsz,2)
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1,nb_nodes):
            current_cities = x[arange_vec, tour[:,i], :] 
            L += torch.sum( (current_cities - previous_cities)**2 , dim=1 )**0.5 # dist(current, previous node) 
            previous_cities = current_cities
        L += torch.sum( (current_cities - first_cities)**2 , dim=1 )**0.5 # dist(last, first node)  
    return L

def evaluate():
    pass
def train_rl():
    L_train_total = 0
    L_base_total = 0
    sum_log_probs_total = []
    for i, (data, done) in enumerate(data_loader):
        model.zero_grad()
        ret = model(data)
        tour, sum_log_probs, probs_cat, new_mems = ret

        with torch.no_grad(): 
            ret_b = baseline(data)
            tour_b = ret_b[0]

        L_train_part = compute_tour_length(tour, data)
        L_base_part = compute_tour_length(tour_b, data)

        loss = model.criterion(L_train_part, L_base_part, sum_log_probs)
        loss.backward()
        model.optimizer.step()

        L_train_total += L_train_part
        L_base_total += L_base_part
        sum_log_probs_total.append(sum_log_probs)
        sum_log_probs_total = torch.cat(sum_log_probs_total, dim=0)

        if args.update_intermediate:
            model.zero_grad()
            loss = model.criterion(L_train_total, L_base_total, sum_log_probs_total)
            loss.backward()
            model.optimizer.step()

        if done:
            if args.update_total:
                model.zero_grad()
                loss = model.criterion(L_train_total, L_base_total, sum_log_probs_total)
                loss.backward()
                model.optimizer.step()
            L_train_total = 0
            L_base_total = 0
            sum_log_probs = []

            if i % args.eval_interval:
                pass

