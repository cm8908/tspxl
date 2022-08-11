class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
args = DotDict()
'''
TSP50
BSZ 512
Attn Type 3 (no PE, no segm)
Integrated version.
'''

# Learning Method #
args.rl = True
args.loss_fn = 'reinforce'
args.optim = 'adam'
args.learning_rate = 0.0001
args.tol = 0.001
args.attn_type = 3

# Experimental #
args.debug = False
args.exp_dir = 'logs/tsp50_att3'
args.seed = 0
args.log_interval = 500

# GPU #
args.cuda = True
args.gpu_id = "6"
args.multi_gpu = False  # currently not compatible with self.parameters()

# Data-Related #
args.data_root = '../datasets'
args.data_source = 'joshi'
args.n_point = 50
args.sorted = False
args.bsz = 512
args.segm_len = 25
args.n_epoch = 10000
args.rl_maxstep = 2500  # nb_batch_per_epoch and eval interval
args.rl_eval_maxstep = 20  # nb_batch_eval

# Model Hyperparameters #
args.d_model = 128
args.d_ff = 512
args.n_head = 8
args.n_enc_layer = 6
args.n_dec_layer = 2

# Minor Hyperparameters #
args.deterministic = False
args.pre_lnorm = True
args.dropout_rate = 0.0
args.internal_drop = -1
args.clip_value = 10
args.clamp_len = -1