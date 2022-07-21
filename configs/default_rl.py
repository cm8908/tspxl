class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
args = DotDict()

# Learning Method #
args.rl = True
args.loss_fn = 'reinforce'
args.update_intermediate = False
args.update_total = True

# Experimental #
args.debug = True
args.exp_dir = 'debug'
args.seed = 1234
args.eval_interval

# GPU #
args.cuda = True
args.gpu_id = '0,1'
args.multi_gpu = False

# Data-Related #
args.data_root = '../datasets'
args.data_source = 'joshi'
args.n_point = 50
args.bsz = 1
args.segm_len = 25

# Optimizer #
args.optim = 'adam'

# Model Hyperparameters #
args.d_model = 1
args.d_ff = 1
args.n_head = 1
args.n_layer = 1

# Minor Hyperparameters #
args.deterministic = True
args.pre_lnorm = True
args.dropout_rate = 0.0
args.internal_drop = -1
args.clip_value = -1
args.clamp_len = -1