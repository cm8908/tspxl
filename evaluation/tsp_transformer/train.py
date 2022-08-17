"""

"""
import importlib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-c', dest='config_name', type=str, default='debug', help='Choose hyperparameter configuration file')
parser.add_argument('-r', dest='restart', type=int, default=-1, help='Epoch of model to restore')
parser.add_argument('--reset-log', dest='reset_log', action='store_true', help='whether to erase log file or not')
cmd_args = parser.parse_args()
config_filename = 'configs.' + cmd_args.config_name
args = importlib.import_module(config_filename).args
print('Loading configurations from', config_filename)
# from configs.default_rl_step import args

import os
if args.cuda:
    assert len(args.gpu_id) > 0
    if args.multi_gpu:
        assert len(args.gpu_id) >= 2
    else:
        assert len(args.gpu_id) == 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import torch
import numpy as np
import pickle

from torch import nn, optim
from time import time
from datetime import datetime
# from models.model import TSPXL, T_ENCODER_LIST, T_DECODER_LIST, T_LOOP_LIST
# from models.decoder import T_DECODER_LOOP_LIST, T_SELFATTN_LIST, T_ATTN_LIST
from model import *
from utils.exp_utils import create_exp_dir, save_checkpoint
from utils.data_utils import RandomTSPGenerator, SortedTSPGenerator, TSPDataset

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
if cmd_args.reset_log:
    log_path = os.path.join(args.exp_dir, 'log.txt')
    os.remove(log_path)
scripts_to_save = ['train.py', 'model.py']
log = create_exp_dir(args.exp_dir, scripts_to_save, args.debug)
log('$' * 100)
log('Program started at ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log('$' * 100)

# Load Dataset and iterator
if args.rl:
    if args.sorted:
        train_loader = SortedTSPGenerator(bsz=args.bsz, total_len=args.n_point, max_step=args.rl_maxstep, device=device, segm_len=args.segm_len)
        val_loader = SortedTSPGenerator(bsz=args.bsz, total_len=args.n_point, max_step=args.rl_eval_maxstep, device=device, segm_len=args.segm_len)
    else:
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
model = TSPNet(
    d_model=args.d_model,
    d_ff=args.d_ff,
    n_head=args.n_head,
    n_enc_layer=args.n_enc_layer,
    n_dec_layer=args.n_dec_layer,
    segm_len=args.segm_len,
    dropout_rate=args.dropout_rate,
    internal_drop=args.internal_drop,
    clip_value=args.clip_value,
    pre_lnorm=args.pre_lnorm,
    attn_type=args.attn_type
)
if args.rl:
    baseline = TSPNet(
    d_model=args.d_model,
    d_ff=args.d_ff,
    n_head=args.n_head,
    n_enc_layer=args.n_enc_layer,
    n_dec_layer=args.n_dec_layer,
    segm_len=args.segm_len,
    dropout_rate=args.dropout_rate,
    internal_drop=args.internal_drop,
    clip_value=args.clip_value,
    pre_lnorm=args.pre_lnorm,
    attn_type=args.attn_type
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
    # baseline = nn.parallel.DistributedDataParallel(model).to(device)
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
             tour of size (N, B) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    x = x.permute(1,0,2).contiguous()  # (B, N, 2)
    tour = tour.permute(1,0).contiguous()  # (B, N)
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

def update_model(tour, tour_b, sum_log_probs, data):
    start = time()
    optimizer.zero_grad()
    L_train = compute_tour_length(tour, data)
    L_base = compute_tour_length(tour_b, data)
    loss = criterion(L_train, L_base, sum_log_probs)
    loss.backward()
    optimizer.step()
    dur = time() - start
    return dur, L_train.mean().item(), L_base.mean().item(), loss.mean().item()

min_tour_length = torch.inf

def eval_rl():
    global min_tour_length

    L_train_list = []
    L_base_list = []
    val_loss_list = []

    eval_start_time = time()

    model.eval()
    baseline.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):

            # tour_list = []
            # tour_b_list = []
            # sum_log_probs_total = 0

            # for j, (data, _) in enumerate(val_loader.get_split_iter(full_data)):
            ret = model(data, True)
            tour, sum_log_probs, probs_cat = ret

            ret_b = baseline(data, True)
            tour_b, _, _ = ret_b
            
            # tour_list.append(tour)
            # tour_b_list.append(tour_b)
            # sum_log_probs_total = sum_log_probs_total + sum_log_probs
            # End of Inner Loop (Full Data) #
            # tour_cat = torch.cat(tour_list, dim=0)
            # tour_b_cat = torch.cat(tour_b_list, dim=0)

            L_train = compute_tour_length(tour, data)
            L_base = compute_tour_length(tour_b, data)
            val_loss = criterion(L_train, L_base, sum_log_probs)
        
            L_train_list.append(L_train.mean().item())
            L_base_list.append(L_base.mean().item())
            val_loss_list.append(val_loss.mean().item())

    L_train_mean = np.mean(L_train_list)
    L_base_mean = np.mean(L_base_list)
    val_loss_mean = np.mean(val_loss_list)

    dur = time() - eval_start_time
    
    # Update baseline if train model is better
    if L_train_mean + args.tol < L_base_mean:
        baseline.load_state_dict(model.state_dict())
        log('@' * 100)
        log('Baseline has been updated. Mean L_train ' +str(L_train_mean))
        log('@' * 100)


    # Save model if shorted tour length
    if not args.debug:
        if min_tour_length > L_train_mean:
            min_tour_length = L_train_mean
            save_checkpoint(model, optimizer, args.exp_dir)
            log('@' * 100)
            log('Model has been saved. Min Mean L_train '+str(min_tour_length))
            log('@' * 100)

            example_path = os.path.join(args.exp_dir, f'example.pt')
            with open(example_path, 'wb') as f:
                example = {
                    'tour': tour, 'tour_b': tour_b,
                    'L_train': L_train, 'L_base': L_base, 'data': data
                }
                torch.save(example, f)
            log(f'Model, Optimizer and Example dictionary are saved at Epoch {e}')

    log('#' * 100)
    log_str = f'Eval Log -- (Step:{args.rl_eval_maxstep}) | Total Time {dur:.3f}s | Time per step {dur/args.rl_eval_maxstep:.3f}s | Mean L_train {L_train_mean:.5f} | Mean L_base {L_base_mean:.5f} | Mean Val Loss {val_loss_mean:.5f}'
    log(log_str)
    log('#' * 100)


def train_rl():
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    t_epoch_start = time()

    t_model_forward_list = []
    t_update_list = []
    # t_update_interm_list = []  # Save intermediate update?
    # t_update_total_list = []

    L_train_track = []
    L_base_track = []
    loss_track = []

    # total_L_train_track = []
    # total_L_base_track = []
    # total_loss_track = []

    model.train()
    baseline.train()
    for i, data in enumerate(train_loader):

        t_one_batch_start = time()
        
        # tour_list = []
        # tour_b_list = []
        # sum_log_probs_list = []
        # sum_log_probs_total = 0
    
        # for j, (data, done) in enumerate(train_loader.get_split_iter(full_data)):
        t_model_forward_start = time()
        '''
        ret:
            tour : (N, B)
            sum_log_probs : (B)
            probs_cat : (N, N, B)
        '''
        ret = model(data, args.deterministic)
        t_model_forward_dur = time() - t_model_forward_start
        t_model_forward_list.append(t_model_forward_dur)
        tour, sum_log_probs, heatmap = ret

        with torch.no_grad(): 
            ret_b = baseline(data, True)
            tour_b, _, _ = ret_b
            
        # Update model using step tour
        # if args.update_step:
        dur, L_train, L_base, loss = update_model(tour, tour_b, sum_log_probs, data)
        t_update_list.append(dur)
        L_train_track.append(L_train)
        L_base_track.append(L_base)
        loss_track.append(loss)


        if (i+1) % args.log_interval == 0:
            t_one_batch = time() - t_one_batch_start
            t_model_forward_mean = np.mean(t_model_forward_list)
            t_update_step = np.mean(t_update_list)
            t_encoder_mean = np.mean(T_ENCODER_LIST)
            t_decoder_mean = np.mean(T_DECODER_LIST)
            t_loop_mean = np.mean(T_LOOP_LIST)
            t_decoder_loop_mean = np.mean(T_DECODER_LOOP_LIST)
            t_selfattn_mean = np.mean(T_SELFATTN_LIST)
            t_attn_mean = np.mean(T_ATTN_LIST)
            # t_update_interm = np.mean(t_update_interm_list)
            # t_update_total = np.mean(t_update_total_list)
            mean_tour_train = np.mean(L_train_track)
            mean_tour_base = np.mean(L_base_track)
            mean_loss_track = np.mean(loss_track)
            # mean_tour_train_total = np.mean(total_L_train_track)
            # mean_tour_base_total = np.mean(total_L_base_track)
            # mean_loss_track_total = np.mean(total_loss_track)

            
            log('#' * 100)
            log_str = f'Train Log -- (Step:{i}) | Batch Duration {t_one_batch:.3f}s | Mean Forward Time {t_model_forward_mean:.3f}'
            log_str += f'\n\tBackward Time {t_update_step:.3f}s | Mean L_train {mean_tour_train:.5f} | Mean L_base {mean_tour_base:.5f} | Mean Train Loss {mean_loss_track:.5f}'
            log_str += f' | Encoder Time {t_encoder_mean} | Decoder Time {t_decoder_mean} | Loop Time {t_loop_mean} | Decoder Loop {t_decoder_loop_mean} | SelfAttn {t_selfattn_mean} | Attn {t_attn_mean}'
            # if args.update_intermediate:
            #     log_str += f'\n\tIntermediate Backward time {t_update_interm:.3f}'
            # if args.update_total:
            #     log_str += f'\n\tTotal Backward Time {t_update_total:.3f} | Mean L_train {mean_tour_train_total:.5f} | Mean L_base {mean_tour_base_total:.5f} | Mean Train Loss {mean_loss_track_total:.5f}'
            log(log_str)
            log('#' * 100)
    # End of Outer Loop (Epoch) #
    t_epoch = time() - t_epoch_start
    eval_rl()
    return t_epoch

# TODO:

def eval_sl():
    pass
def train_sl():
    pass

if cmd_args.restart > 0:
    log(f'Restarting from epoch {cmd_args.restart}')
    model_to_load = os.path.join(args.exp_dir, f'model.pt')
    opt_to_load = os.path.join(args.exp_dir, f'optimizer.pt')
    with open(model_to_load, 'rb') as m, open(opt_to_load, 'rb') as o:
        model = torch.load(m)
        optimizer.load_state_dict(torch.load(o))
        print(type(model), type(optimizer))
    start_epoch = cmd_args.restart + 1
else:
    start_epoch = 0
try:
    t_train_start = time()
    for e in range(start_epoch, args.n_epoch):
        t_epoch = train_rl()
        log('@' * 100)
        log(f'Epoch {e} has been ended. Duration: {t_epoch:.3f}s')
        log('@' * 100)
    t_train = time() - t_train_start
    log('$' * 100)
    log(f'Train has been ended. Duration: {t_train:.3f}s')
    log('$' * 100)
    
except KeyboardInterrupt:
    log('Training has been stopped early')