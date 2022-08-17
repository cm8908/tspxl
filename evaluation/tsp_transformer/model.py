T_ATTN_LIST, T_SELFATTN_LIST, T_DECODER_LOOP_LIST, T_DECODER_LIST, T_ENCODER_LIST, T_LOOP_LIST = [], [], [], [], [], []
import torch
from torch import nn
from torch.distributions import Categorical
from time import time

class TSPEncoder(nn.Module):
    """
    Encodes 2-D Euclidean TSP instance into context tensor by applying MHSA
    Notes:
        N : number of points in current segment (=L_seg)
        B : batch size
        H : hidden dimension
    """
    def __init__(self, d_model, d_ff, n_head, n_layer, dropout_rate=0.2):
        ''' docstring '''
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head

        self.MHA_layers = nn.ModuleList([nn.MultiheadAttention(d_model, n_head) for _ in range(n_layer)])
        self.FFN_layers_1 = nn.ModuleList([nn.Linear(d_model, d_ff) for _ in range(n_layer)])
        self.FFN_layers_2 = nn.ModuleList([nn.Linear(d_ff, d_model) for _ in range(n_layer)])
        self.BN_layers_1 = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(n_layer)])
        self.BN_layers_2 = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(n_layer)])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h):
        '''
        Inputs:
            h : Input embedding + start token  # size=(N+1, B, H)
        Outputs:
            h : Encoded context vector  # size=(N+1, B, H)
            attn_weights : Attention weights  # size=(B, N+1, N+1)
        '''
        for l in range(self.n_layer):
            h_rc = h
            # Multi-Head Attention
            h, _ = self.MHA_layers[l](h, h, h)
            
            # Add & Norm
            h = h_rc + h
            h = h.permute(1,2,0).contiguous()  # (B, H, N+1)
            h = self.BN_layers_1[l](h)
            h = h.permute(2,0,1).contiguous()  # (N+1, B, H)

            h = self.dropout(h)

            h = h_rc
            # FFN layer
            h = self.FFN_layers_1[l](h)
            h = torch.relu(h)
            h = self.FFN_layers_2[l](h)

            # Add & Norm
            h = h_rc + h
            h = h.permute(1,2,0).contiguous()
            h = self.BN_layers_2[l](h)
            h = h.permute(2,0,1).contiguous()
            
            h = self.dropout(h)

            return h

def rel_shift(x):
    zero_pad = torch.zeros(x.size(0), 1, *x.size()[2:], device=x.device, dtype=x.dtype)
    x_pad = torch.cat([zero_pad, x], dim=1)
    x_pad = x_pad.view(x.size(1) + 1, x.size(0), *x.size()[2:])
    x = x_pad[1:].view_as(x)
    return x

class Attention(nn.Module):
    def __init__(self, n_head, d_model, attn_type=0, clip_value=None):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_model//n_head
        self.clip = clip_value

        if attn_type == 2:
            self.r_net = nn.Linear(d_model, d_model)
    
    def forward(self, q, K, V, mask=None, rel=None):
        '''
        q : (qlen, B, H)
        K : (klen, B, H)
        V : (klen, B, H)
        mask : (1, N, B, 1)
        '''
        qlen, bsz, klen = q.size(0), q.size(1), K.size(0)
        q = q.view(qlen, bsz, self.n_head, self.d_head)  # (qlen, B, nh, D)
        K = K.view(klen, bsz, self.n_head, self.d_head)  # (klen, B, nh, D)
        V = V.view(klen, bsz, self.n_head, self.d_head)  # (klen, B, nh, D)
        if rel is not None:
            r, u, v = rel
            r = self.r_net(r)
            r = r.view(klen, self.n_head, self.d_head)  # (klen, nh, D)
            AC = torch.einsum('ibnd,jbnd->ijbn', q+u, K)  # (1, N, B, nh)
            BD = torch.einsum('ibnd,jnd->ijbn', q+v, r)  # (1, N, B, nh)
            BD = rel_shift(BD)
            score = (AC + BD) / self.d_head**0.5  # (1, N, B, nh)
        else:
            score = torch.einsum('ibnd,jbnd->ijbn', q, K) / self.d_head**0.5  # (1, N, B, nh)
        if self.clip is not None:
            score = self.clip * torch.tanh(score)  # (1, N, B, nh)
        if mask is not None:
            mask = mask.repeat(1, 1, 1, self.n_head)  # (1, N, B, nh)
            score.masked_fill_(mask, float('-1e9'))
        weight = torch.softmax(score, dim=1)  # (1, N, B, nh)
        out = torch.einsum('ijbn,jbnd->ibnd', weight, V)  # (1, B, nh, D)
        out = out.view(qlen, bsz, self.n_head*self.d_head)  # (1, B, H)
        weight = weight.mean(dim=-1)  # (1, N, B)
        return out, weight

class DecoderLayer(nn.Module):
    """
    Perform attention across segments (+ position-wise FFN)
    attn_type = 0 (Bresson), 1 (BressonXL), 2 (BressonXL+RPE)
    """
    def __init__(self, d_model, n_head, dropout_rate, internal_drop, pre_lnorm, segm_len, attn_type=0):
        super().__init__()

        self.Wq_sa = nn.Linear(d_model, d_model)
        self.Wk_sa = nn.Linear(d_model, d_model)
        self.Wv_sa = nn.Linear(d_model, d_model)
        self.W0_sa = nn.Linear(d_model, d_model)
        self.K_sa = None
        self.V_sa = None

        self.Wq_a = nn.Linear(d_model, d_model)
        self.W0_a = nn.Linear(d_model, d_model)

        # self.MHSA = nn.MultiheadAttention(d_model, n_head)
        self.SelfAttn = Attention(n_head, d_model, internal_drop, attn_type)
        self.EncDecAttn = Attention(n_head, d_model, internal_drop)
        if pre_lnorm:
            self.prenorm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.ffn1 = nn.Linear(d_model, d_model)
        self.ffn2 = nn.Linear(d_model, d_model)

        self.segm_len = segm_len
        self.attn_type = attn_type

    def forward(self, h_t, K_a, V_a, mask=None, rel=None):
        """

        """
        # Self-Attention Weights
        q_sa = self.Wq_sa(h_t)  # (1, B, H)
        k_sa = self.Wk_sa(h_t)  # (1, B, H)
        v_sa = self.Wv_sa(h_t)  # (1, B, H)
        
        if self.K_sa is None:
            self.K_sa = k_sa
            self.V_sa = v_sa
        else:
            self.K_sa = torch.cat([self.K_sa, k_sa], dim=0)  # (1~N B, H)
            self.V_sa = torch.cat([self.V_sa, k_sa], dim=0)  # (1~N B, H)
        if self.attn_type == 1 or self.attn_type == 2:
            self.K_sa = self.K_sa[-self.segm_len:, :, :]
            self.V_sa = self.V_sa[-self.segm_len:, :, :]

        # Multi-Head Self-Attention
        t_selfattn_start = time()
        out, _ = self.SelfAttn(q_sa, self.K_sa, self.V_sa, rel=rel)  # size(out)=(1, B, H)
        out = self.W0_sa(out)  # (1, B, H)
        t_selfattn = time() - t_selfattn_start
        T_SELFATTN_LIST.append(t_selfattn)

        # Add & Norm
        h_t = h_t + out  # (1, B, H)
        h_t = self.norm1(h_t)  # (1, B, H)

        h_t = self.dropout(h_t)
       
        # Multi-Head Attention Weights
        try: h_t = self.prenorm(h_t)
        except AttributeError: pass
        q_a = self.Wq_a(h_t)  # (1, B, H)

        # Encoder - Decoder Attention
        t_attn_start = time()
        out, _ = self.EncDecAttn(q_a, K_a, V_a, mask)
        out = self.W0_a(out)
        t_attn = time() - t_attn_start
        T_ATTN_LIST.append(t_attn)

        # Add & Norm
        h_t = h_t + out
        h_t = self.norm2(h_t)

        h_t = self.dropout(h_t)

        # FFN
        out = self.ffn1(h_t)
        out = torch.relu(out)
        out = self.ffn2(out)
        
        # Add & Norm
        h_t = h_t + out
        h_t = self.norm3(h_t)

        return h_t  # (1, B, H)


class TSPDecoder(nn.Module):
    """
    Decodes each token at a time
        For i=0 to L_seg-1:
        Q = H_enc[i]
        K, V = H_enc[:]
        Decode Q,K,V into probability matrix of size=(B, L_seg, N)
        Select i-th city out of W, mask visited one, and store selected log prob
    """
    def __init__(self, d_model, n_layer, n_head, dropout_rate, internal_drop, clip_value, pre_lnorm, segm_len, attn_type=0):
        super().__init__()

        self.d_model = d_model
        self.n_layer = n_layer

        self.Wq_final = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, dropout_rate, internal_drop, pre_lnorm, segm_len, attn_type) for _ in range(n_layer)])
        self.sha = Attention(1, d_model, internal_drop, clip_value)

        self.attn_type = attn_type

    def reset_KV_sa(self):
        for layer in self.layers:
            layer.K_sa = None
            layer.V_sa = None

    def forward(self, h_t, K_a, V_a, mask=None, rel=None):
        """
        Inputs:
            h_t : t-th token of current segment  # size=(1, B, H)
            K_a,V_a : key for enc-dec attention  # size=(N+1, B, )
            mask : (1, N, B, 1)
            rel : tuple(r, u, v)
        """

        # Loop for each decoder layer
        t_decoder_loop_start = time()
        for i in range(self.n_layer):
            K_a_l = K_a[:,:,i*self.d_model:(i+1)*self.d_model].contiguous()
            V_a_l = V_a[:,:,i*self.d_model:(i+1)*self.d_model].contiguous()
            if i < self.n_layer - 1:
                h_t = self.layers[i](h_t, K_a_l, V_a_l, mask, rel)  # (1, B, H)
            else:
                q_final = self.Wq_final(h_t)  # (1, B, H)
                _, probs = self.sha(q_final, K_a_l, V_a_l, mask)  # (1, N, B)
        t_decoder_loop = time() - t_decoder_loop_start
        T_DECODER_LOOP_LIST.append(t_decoder_loop)
        
        return probs

def generate_positional_encoding(pos_seq, d_model):
    div_term = torch.exp(torch.arange(0, d_model, 2).to(pos_seq.device).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    sinusoid = torch.outer(pos_seq, div_term)
    pe = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=1)
    return pe[:,None,:]

class TSPNet(nn.Module):
    def __init__(self, d_model, d_ff, n_head, n_enc_layer, n_dec_layer, segm_len, dropout_rate, internal_drop, clip_value, pre_lnorm, maxlen=1000, attn_type=0):
        super().__init__()

        self.W_kv_decoder = nn.Linear(d_model, n_dec_layer*d_model*2)
        self.input_emb = nn.Linear(2, d_model)
        self.start_tokens = nn.Parameter(torch.randn(d_model))
        self.encoder = TSPEncoder(d_model, d_ff, n_head, n_enc_layer, dropout_rate)
        self.decoder = TSPDecoder(d_model, n_dec_layer, n_head, dropout_rate, internal_drop, clip_value, pre_lnorm, segm_len, attn_type)

        self.attn_type = attn_type
        if attn_type == 0 or attn_type == 1:
            pos_seq = torch.arange(maxlen)
            self.ape = generate_positional_encoding(pos_seq, d_model)
        if attn_type == 2:
            self.u = nn.Parameter(torch.Tensor(n_head, d_model//n_head))
            self.v = nn.Parameter(torch.Tensor(n_head, d_model//n_head))
            self.segm_len = segm_len
    def forward(self, x, deterministic=False):
        '''
        x : (N, B, 2)
        '''
        N, bsz = x.size()[:2]
        toB = torch.arange(bsz)

        h = self.input_emb(x)  # (N, B, H)
        h = torch.cat([h, self.start_tokens.repeat(1,bsz,1)], dim=0)  # (N+1, B, H)

        # Encode it !
        t_encoder_start = time()
        h_enc = self.encoder(h)  # (N+1, B, H)
        t_encoder = time() - t_encoder_start
        T_ENCODER_LIST.append(t_encoder)

        # Start token
        h_start = h_enc[-1:, toB, :]  # (1, B, H)

        # Track lists
        tour = []
        probs_cat = []
        log_probs = []

        # Initializing mask
        mask = torch.zeros(1, N+1, bsz, 1, device=x.device).bool()
        mask[:, -1, :, :] = True

        #
        rel = None
        if self.attn_type == 0 or self.attn_type == 1:
            self.ape = self.ape.to(x.device)

        # Decode it !
        h_t = h_start
        KV_a = self.W_kv_decoder(h_enc)
        K_a, V_a = torch.chunk(KV_a, 2, dim=-1)  # (N+1, B, H*n_dec_layer)
        self.decoder.reset_KV_sa()

        # Decoding Loop
        t_loop_start = time()
        for t in range(N):

            if self.attn_type == 0 or self.attn_type == 1:
                h_t = h_t + self.ape[t].repeat(1,bsz,1)
            if self.attn_type == 2:
                pos_seq = torch.arange(min(t, self.segm_len-1), -1, -1, device=h_t.device, dtype=h_t.dtype)
                r = generate_positional_encoding(pos_seq, h_t.size(-1))
                rel = (r, self.u, self.v)

            # Decode one step
            t_decoder_start = time()
            probs = self.decoder(h_t, K_a, V_a, mask, rel)
            t_decoder = time() - t_decoder_start
            T_DECODER_LIST.append(t_decoder)

            # Choose city
            if deterministic:
                city = probs.argmax(dim=1)  # (1, B)
            else:
                city = Categorical(probs.permute(0,2,1)).sample()  # (1, B)
            
            city_idx = city.squeeze()  # (B)

            tour.append(city)
            probs_cat.append(probs)
            chosen_prob = probs[:, city_idx, toB]  # (1, B)
            log_probs.append(chosen_prob.log())

            # update mask
            if mask is not None:
                mask = mask.clone()
                mask[:, city_idx, toB, :] = True

            h_t = h_enc[city_idx, toB, :].unsqueeze(0)
        t_loop = time() - t_loop_start
        T_LOOP_LIST.append(t_loop)

        tour = torch.cat(tour, dim=0)  # (N, B)
        sum_log_probs = torch.cat(log_probs, dim=0).sum(dim=0)  # (B)
        probs_cat = torch.cat(probs_cat, dim=0)  # (N, N, B)
            
        return tour, sum_log_probs, probs_cat  #, loss

START = None
LONG = 20
def check(flag):
    global START
    if flag == 0:
        if START != None:
            START = None
            raise Exception('Run the cell again')
        START = time()
    elif flag == 1:
        print(time() - START)
        START = None
    else:
        raise Exception('Flag must be given')

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    d_model, d_ff, n_head, n_enc_layer, n_dec_layer = 128, 512, 8, 6, 2
    segm_len, dropout_rate, internal_drop, clip_value, pre_lnorm = 25, 0.2, -1, 10, False

    N, B = 50, 10
    sample = torch.rand(N, B, 2).cuda()

    check(0)
    model_0 = TSPNet(d_model, d_ff, n_head, n_enc_layer, n_dec_layer, segm_len,
                     dropout_rate, internal_drop, clip_value, pre_lnorm, attn_type=0).cuda()
    
    for _ in range(LONG):
        tour, _, _ = model_0(sample)
    check(1)

    check(0)
    model_1 = TSPNet(d_model, d_ff, n_head, n_enc_layer, n_dec_layer, segm_len,
                     dropout_rate, internal_drop, clip_value, pre_lnorm, attn_type=1).cuda()
    for _ in range(LONG):
        tour, _, _ = model_1(sample)
    check(1)

    check(0)
    model_2 = TSPNet(d_model, d_ff, n_head, n_enc_layer, n_dec_layer, segm_len,
                     dropout_rate, internal_drop, clip_value, pre_lnorm, attn_type=2).cuda()
    for _ in range(LONG):
        tour, _, _ = model_2(sample)
    check(1)

    check(0)
    model_3 = TSPNet(d_model, d_ff, n_head, n_enc_layer, n_dec_layer, segm_len,
                     dropout_rate, internal_drop, clip_value, pre_lnorm, attn_type=3).cuda()
    for _ in range(LONG):
        tour, _, _ = model_3(sample)
    check(1)
    
