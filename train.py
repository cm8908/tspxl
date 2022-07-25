"""

"""
from configs.default_rl import args
import os
if args.cuda:
    assert len(args.gpu_id) > 0
    if args.multi_gpu:
        assert len(args.gpu_id) >= 2
    else:
        assert len(args.gpu_id) == 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import shutil
import torch
import numpy as np

from torch import nn, optim
from time import time
from models.model import TSPXL
from utils.exp_utils import create_exp_dir
from utils.data_utils import RandomTSPGenerator, TSPDataset

# Set seed and cuda
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda and torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Total {torch.cuda.device_count()} GPUs in USE')
    print(f'<< GPU : {torch.cuda.get_device_name()}, id: {args.gpu_id} >>')
else:
    device = torch.device('cpu')

# Set logger
scripts_to_save = ['train.py', 'models']
log = create_exp_dir(args.exp_dir, scripts_to_save, args.debug)

# Load Dataset and iterator
if args.rl:
    train_loader = RandomTSPGenerator(bsz=args.bsz, total_len=args.n_point, max_step=args.rl_maxstep, device=device, segm_len=args.segm_len)
    val_loader = RandomTSPGenerator(bsz=args.bsz, total_len=args.n_point, max_step=args.rl_eval_maxstep, device=device, segm_len=args.segm_len)
else:
    if args.n_point >= 50:
        train_set = TSPDataset(n=args.n_point, mode='train', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        test_set = TSPDataset(n=args.n_point, mode='test', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        train_set, val_set = train_set[:-len(test_set)], train_set[-len(test_set):]
    else:
        train_set = TSPDataset(n=args.n_point, mode='train', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        val_set = TSPDataset(n=args.n_point, mode='val', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
        test_set = TSPDataset(n=args.n_point, mode='test', root_dir=args.data_root, author=args.data_source, device=device, segm_len=args.segm_len)
    
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

# Build model
model = TSPXL(
    d_model=args.d_model,
    d_ff=args.d_ff,
    n_head=args.n_head,
    n_enc_layer=args.n_enc_layer,
    n_dec_layer=args.n_dec_layer,
    n_class=args.n_point,
    bsz=args.bsz,
    deterministic=args.deterministic,
    criterion=criterion,
    dropout_rate=args.dropout_rate,
    internal_drop=args.internal_drop,
    clip_value=args.clip_value,
    pre_lnorm=args.pre_lnorm,
    clamp_len=args.clamp_len,
    rl=args.rl
)
if args.rl:
    baseline = TSPXL(
    d_model=args.d_model,
    d_ff=args.d_ff,
    n_head=args.n_head,
    n_enc_layer=args.n_enc_layer,
    n_dec_layer=args.n_dec_layer,
    n_class=args.n_point,
    bsz=args.bsz,
    deterministic=args.deterministic,
    criterion=criterion,
    dropout_rate=args.dropout_rate,
    internal_drop=args.internal_drop,
    clip_value=args.clip_value,
    pre_lnorm=args.pre_lnorm,
    clamp_len=args.clamp_len,
    rl=args.rl
    )

# Optimizer
if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
else:
    raise ValueError("Proper optimizer should be provided")

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
    Original code from : github.com/xbresson
    Compute the length of a batch of tours
    Inputs : x of size (N, B, 2) batch of tsp tour instances
             tour of size (B, N) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    x = x.permute(1,0,2).contiguous()  # (B, N, 2)
    bsz = x.shape[0]
    nb_nodes = tour.shape[1]
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

def update_model(tour, tour_b, sum_log_probs, full_data):
    optimizer.zero_grad()
    L_train = compute_tour_length(tour, full_data)
    L_base = compute_tour_length(tour_b, full_data)
    loss = model.criterion(L_train, L_base, sum_log_probs)
    loss.backward()
    optimizer.step()

def eval_rl():

    L_train_list = []
    L_base_list = []
    val_loss_list = []

    eval_start_time = time()

    model.eval()
    baseline.eval()
    with torch.no_grad():
        for i, full_data in enumerate(val_loader):

            tour_list = []
            tour_b_list = []
            sum_log_probs_total = 0

            mask = torch.zeros(args.bsz, args.d_model, args.n_point, device=device, dtype=torch.bool)
            mask_b = torch.zeros(args.bsz, args.d_model, args.n_point, device=device, dtype=torch.bool)
            mems = tuple()
            mems_b = tuple()

            for j, data in enumerate(val_loader.get_split_iter(full_data)):
                ret = model(data, None, mask, *mems)
                tour, sum_log_probs, probs_cat, mask, mems = ret

                ret_b = baseline(data, None, mask_b, *mems_b)
                tour_b, mask_b, mems_b = ret_b[0], ret_b[3], ret_b[4]
            
                tour_list.append(tour)
                tour_b_list.append(tour_b)
                sum_log_probs_total = sum_log_probs_total + sum_log_probs
            # End of Inner Loop (Full Data) #
            tour_cat = torch.cat(tour_list, dim=1)
            tour_b_cat = torch.cat(tour_b_list, dim=1)

            L_train = compute_tour_length(tour_cat, full_data)
            L_base = compute_tour_length(tour_b_cat, full_data)
            val_loss = model.criterion(L_train, L_base, sum_log_probs_total)
        
        L_train_list.append(L_train.mean().item())
        L_base_list.append(L_base.mean().item())
        val_loss_list.append(val_loss.mean().item())

    L_train_mean = np.mean(L_train_list)
    L_base_mean = np.mean(L_base_list)
    val_loss_mean = np.mean(val_loss_list)

    dur = time() - eval_start_time

    log('#' * 100)
    log_str = f'Eval ({args.rl_eval_maxstep}) - Time {dur}s | Mean L_train {L_train_mean} | Mean L_base {L_base_mean} | Mean Valid Loss {val_loss_mean}'
    log(log_str)
    log('#' * 100)


def train_rl():
    torch.autograd.set_detect_anomaly(True)

    model.train()
    baseline.train()
    for i, full_data in enumerate(train_loader):
        
        tour_list = []
        tour_b_list = []
        sum_log_probs_total = 0
    
        mask = torch.zeros(args.bsz, args.d_model, args.n_point, device=device, dtype=torch.bool)
        mask_b = torch.zeros(args.bsz, args.d_model, args.n_point, device=device, dtype=torch.bool)
        mems = tuple()
        mems_b = tuple()

        for j, data in enumerate(train_loader.get_split_iter(full_data)):
            ret = model(data, None, mask, *mems)
            tour, sum_log_probs, probs_cat, mask, mems = ret

            with torch.no_grad(): 
                ret_b = baseline(data, None, mask_b, *mems_b)
                tour_b, mask_b, mems_b = ret_b[0], ret_b[3], ret_b[4]
            
            # Update model using step tour
            if args.update_step:
                update_model(tour, tour_b, sum_log_probs, full_data)

            tour_list.append(tour)
            tour_cat = torch.cat(tour_list, dim=1)

            tour_b_list.append(tour_b)
            tour_b_cat = torch.cat(tour_b_list, dim=1)

            sum_log_probs_total = sum_log_probs_total + sum_log_probs

            # Update model using intermediate tour
            if args.update_intermediate and j != 0:
                update_model(tour_cat, tour_b_cat, sum_log_probs_total, full_data)
            

        # End of Inner Loop (Full Data) #
        # Update model using complete tour
        if args.update_total and not args.update_intermediate:
            update_model(tour_cat, tour_b_cat, sum_log_probs_total, full_data)

        if i % args.log_interval == 0:
            #TODO:
            pass
    # End of Outer Loop (Epoch) #
    eval_rl()

for e in range(args.n_epoch):
    train_rl()