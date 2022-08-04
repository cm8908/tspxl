import torch
from torch import nn
from models.attention import MultiHeadSelfAttn, RelMultiHeadAttn

class PositionalEmbedding(nn.Module):
    """
    Original code from : github.com/kimiyoung
    """
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.drop(torch.relu(self.l1(x)))
        x = self.drop(self.l2(x))
        return x

class DecoderLayer(nn.Module):
    """
    Perform attention across segments (+ position-wise FFN)
    """
    def __init__(self, d_model, d_ff, n_head, dropout_rate, internal_drop, clip_value, pre_lnorm):
        super().__init__()

        self.Wq_sa = nn.Linear(d_model, d_model)
        self.Wk_sa = nn.Linear(d_model, d_model)
        self.Wv_sa = nn.Linear(d_model, d_model)
        self.W0_sa = nn.Linear(d_model, d_model)
        # self.K_sa = None
        # self.V_sa = None

        self.Wq_a = nn.Linear(d_model, d_model)
        self.Wk_a = nn.Linear(d_model, d_model)
        self.Wv_a = nn.Linear(d_model, d_model)
        self.Wr_a = nn.Linear(d_model, d_model)
        self.W0_a = nn.Linear(d_model, d_model)

        # self.MHSA = nn.MultiheadAttention(d_model, n_head)
        self.MHSA = MultiHeadSelfAttn(n_head, d_model, internal_drop, clip_value)
        self.RMHA = RelMultiHeadAttn(n_head, d_model, internal_drop, clip_value)
        if pre_lnorm:
            self.prenorm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.FFN = FFN(d_model, d_ff, dropout_rate)
        # self.FFN = nn.Sequential(
        #     nn.Linear(d_model, d_ff), nn.ReLU(inplace=True),
        #     nn.Dropout(dropout_rate), nn.Linear(d_ff, d_model),
        #     nn.Dropout(dropout_rate)
        # )
        pass

    def forward(self, h_t, h_enc, r, bias_u, bias_v, mask, mem):
        """
        Inputs:
            h_t : token in query  # size=(1, B, H)
            h_enc : keys and values in current segment  # size=(N, B, H)
            r : relative PE  # size=(N, 1, H)
            biases : u, v in Dai et al. 2019  # size=(nh, D)
            mem : keys and values of current layer  # size=(0~N, B, H) where N = L_seg
        Outputs:
            h_t : (1, B, H)
        """
        # Self-Attention Weights
        q_sa = self.Wq_sa(h_t)  # (1, B, H)
        k_sa = self.Wk_sa(h_t)  # (1, B, H)
        v_sa = self.Wv_sa(h_t)  # (1, B, H)
        if mem is None:
            K_sa = k_sa
            V_sa = v_sa
        else:
            K_sa = torch.cat([mem, k_sa], dim=0)
            V_sa = torch.cat([mem, v_sa], dim=0)
        
        # >>> Legacy <<<
        # if self.K_sa is None:
        #     self.K_sa = k_sa
        #     self.V_sa = v_sa
        # else:
        #     self.K_sa = torch.cat([self.K_sa, k_sa], dim=0)  # (1~N B, H)
        #     self.V_sa = torch.cat([self.V_sa, k_sa], dim=0)  # (1~N B, H)
        
        # Multi-Head Self-Attention
        out = self.MHSA(q_sa, K_sa, V_sa)  # size(out)=(1, B, H)
        out = self.W0_sa(out)  # (1, B, H)

        # Add & Norm
        h_t = h_t + out  # (1, B, H)
        h_t = self.norm1(h_t)  # (1, B, H)

        h_t = self.dropout(h_t)
       
        # Multi-Head Attention Weights
        try: h_t = self.prenorm(h_t)
        except: pass
        q_a = self.Wq_a(h_t)  # (1, B, H)
        r_a = self.Wr_a(r)  # (N, 1, H)
        K_a = self.Wk_a(h_enc)  # (N, B, H)
        V_a = self.Wv_a(h_enc)  # (N, B, H)
        
        # >>> Legacy <<<
        # if mem is not None:
        #     cat = torch.cat([mem, h_t], dim=0)
        #     K_a = self.Wk_a(cat)  # (1~N, B, H)
        #     V_a = self.Wv_a(cat)  # (1~N, B, H)
        # else:
        #     K_a = self.Wk_a(h_t)  # (1, B, H)
        #     V_a = self.Wv_a(h_t)  # (1, B, H)

        # Relative Multi-Head Attention
        out, probs = self.RMHA(q_a, K_a, V_a, r_a, bias_u, bias_v, mask)
        out = self.W0_a(out)

        # Add & Norm
        h_t = h_t + out
        h_t = self.norm2(h_t)

        h_t = self.dropout(h_t)

        # FFN
        out = self.FFN(h_t)
        
        # Add & Norm
        h_t = h_t + out
        h_t = self.norm3(h_t)

        return h_t, probs  # (1, B, H)

class TSPDecoder(nn.Module):
    """
    Decodes each token at a time
        For i=0 to L_seg-1:
        Q = H_enc[i]
        K, V = H_enc[:]
        Decode Q,K,V into probability matrix of size=(B, L_seg, N)
        Select i-th city out of W, mask visited one, and store selected log prob
    """
    def __init__(self, d_model, d_ff, n_layer, n_head, segm_len, clamp_len, bsz, dropout_rate, internal_drop, clip_value, pre_lnorm):
        
        assert (internal_drop > 0 and internal_drop < 1) or internal_drop == -1, 'Dropout ratio must be in range (0,1). -1 to be disabled'
        assert (clip_value > 0) or clip_value == -1, 'Clip value must be bigger than 0. Set -1 to be disabled'
        assert d_model % n_head == 0, 'd_model should be dividable by n_head'

        super().__init__()

        self.n_layer = n_layer
        self.clamp_len = clamp_len

        d_head = d_model // n_head
        self.u = nn.Parameter(torch.randn(n_head, d_head))
        self.v = nn.Parameter(torch.randn(n_head, d_head))

        self.pos_emb = PositionalEmbedding(d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_head, dropout_rate, internal_drop, clip_value, pre_lnorm) for _ in range(n_layer)])
        self.classifier = nn.Linear(d_model, segm_len)

    # def reset_KV_sa(self):
    #     for layer in self.layers:
    #         layer.K_sa = None
    #         layer.V_sa = None

    def forward(self, h_t, h_enc, mask, *mems):
        """
        Inputs:
            h_t : t-th token of current segment  # size=(1, B, H)
            mems : list of memory  # size(each mem)=(0~N,B,H)
            mask : mask for visited city same size with classifier # size=(1, N, B)
        """
        # Preparing positional encoding
        qlen = h_t.size(0)
        mlen = mems[0].size(0) if mems is not None else 0
        # klen = mlen + h_enc.size(0)
        klen = mlen + qlen

        pos_seq = torch.arange(h_enc.size(0)-1, -1, -1.0, device=h_t.device, dtype=h_t.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        r = self.pos_emb(pos_seq)
            
        # Drop h_t and r?

        # Loop for each decoder layer
        hids = []
        hids.append(h_t)
        for i in range(self.n_layer):
            mem = mems[i] if mems is not None else None
            h_t, probs = self.layers[i](h_t, h_enc, r, self.u, self.v, mask, mem)  # (1, B, H), (1, N, B)
            hids.append(h_t)


        # update memory
        # self._update_mems(hids, mems)

        return probs, hids, mems


if __name__ == '__main__':
    n_class, B, H = 50, 100, 128
    n_layer = 8
    h_t = torch.randn(1, B, H)
    # mems = [torch.randn(n_class, B, H) for _ in range(n_layer + 1)]
    mems = [torch.empty(0) for _ in range(n_layer+1)]

    decoder = TSPDecoder(d_model=H, d_ff=512, n_layer=n_layer, n_head=8, n_class=n_class, clamp_len=-1, bsz=B, dropout_rate=0.1, internal_drop=-1, clip_value=-1, pre_lnorm=True)

    ret = decoder(h_t, mask=None, mems=mems)
    probs, hids, mems = ret
    print('probs:', probs.shape)
    print('hids:',len(hids), hids[0].shape)
    print('mems:',len(mems), mems[0].shape)