class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
args = DotDict()

# Learning Method #
args.rl = True
args.loss_fn = 'reinforce'
args.optim = 'adam'
args.learning_rate = 0.001
args.update_step = False
args.update_intermediate = False
args.update_total = True

# Experimental #
args.debug = True
args.exp_dir = 'debug'
args.seed = 1234
args.log_interval = 1

# GPU #
args.cuda = True
args.gpu_id = "7"
args.multi_gpu = False

# Data-Related #
args.data_root = '../datasets'
args.data_source = 'joshi'
args.n_point = 50
args.bsz = 512
args.segm_len = 25
args.n_epoch = 10000
args.rl_maxstep = 2500  # batch per epoch
args.rl_eval_maxstep = 20

# Model Hyperparameters #
args.d_model = 128
args.d_ff = 512
args.n_head = 8
args.n_enc_layer = 6
args.n_dec_layer = 2

# Minor Hyperparameters #
args.deterministic = True
args.pre_lnorm = True
args.dropout_rate = 0.1
args.internal_drop = -1
args.clip_value = 10
args.clamp_len = -1