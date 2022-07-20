class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
args = DotDict()

# Experimental #
args.debug = True
args.exp_dir = 'debug'
args.seed = 1234

# GPU #
args.cuda = True
args.gpu_id = '0,1'
args.multi_gpu = False

# Data-Related #
args.data_root = '../datasets'
args.data_source = 'joshi'
args.n_point = 50
args.bsz = 1

# Optimizer #
args.optim = str()

# Loss function #
args.loss_fn = str()

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