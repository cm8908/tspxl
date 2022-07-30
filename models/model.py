import torch
from torch import nn
from torch.distributions import Categorical

from .encoder import TSPEncoder
from .decoder import TSPDecoder

class TSPXL(nn.Module):
    """
    Encoder:
        H_enc = Encoder(x)  # Encodes segmented TSP instance into context vectors with MHSA and FFN layers
    Decoder:
        Decodes autoregressively for L_seg tokens

    """
    def __init__(self, rl, d_model, d_ff, n_head, n_enc_layer, n_dec_layer, segm_len, bsz, deterministic, criterion,
                       dropout_rate, internal_drop, clip_value, pre_lnorm, clamp_len):
        super().__init__()
        self.d_model = d_model
        self.n_enc_layer = n_enc_layer
        self.n_dec_layer = n_dec_layer
        self.deterministic = deterministic
        self.rl = rl


        self.input_emb = nn.Linear(2, d_model)
        self.start_tokens = nn.Parameter(torch.randn(d_model))
        self.encoder = TSPEncoder(d_model, d_ff, n_head, n_enc_layer)
        self.decoder = TSPDecoder(d_model, d_ff, n_dec_layer, n_head, segm_len, clamp_len,
                                  bsz, dropout_rate, internal_drop, clip_value, pre_lnorm)

        self.criterion = criterion
    
    def _init_mems(self):
        mems = []
        param = next(self.parameters())
        for i in range(self.n_dec_layer + 1):
            empty = torch.empty(0, dtype=param.dtype, device=param.device)
            mems.append(empty)
        return mems

    def _update_mems(self, hids, mems, segment_len):
        if mems is None: return None

        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        with torch.no_grad():
            new_mems = []
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)  # (1~N, B, H)
                start_idx = min(segment_len, cat.size(0))
                new_mems.append(cat[-start_idx:].detach())
        return new_mems

    def forward(self, x, target, *mems):
        """
        Inputs
            x : 2D Euclidean TSP instance (segmented data)  # size=(N, B, 2)
            target :   # size=(B, N)
            mask : Initially zeros(B, H, N), keep being updated in `for t loop`
            mems : 
                Initially tuple()
                mems will be updated along `for t loop`, maintaining segment length
        Outputs
        
        """
        segment_len = x.size(0)
        bsz = x.size(1)
        toB = torch.arange(bsz)
        
        h = self.input_emb(x)  # (N, B, H)

        h = torch.cat([h, self.start_tokens.repeat(1,bsz,1)], dim=0)  # (N+1, B, H)

        # Concat with start token
        # h = torch.cat([h, self.start_tokens.repeat(bsz, 1, 1)], dim=0)  # (N+1, B, H)

        # Encode it !
        h_enc, _ = self.encoder(h)  # (N, B, H)

        # Start token
        h_start = h_enc[-1:, toB, :]  # FIXME:

        tour = []
        probs_cat = []
        log_probs = []
        # Decode it !
        self.decoder.reset_KV_sa()
        mask = torch.zeros(1, bsz, segment_len, device=x.device).bool()
        new_mems = []
        h_t = h_start
        for t in range(segment_len):
            # h_t = h_enc[t:t+1]  

            if not mems: mems = self._init_mems()
            '''
            probs : (1, B, N)
            hids, mems : len=n_dec_layer+1, size(hid)=(1,B,H), size(mem)=(N,B,H)
            '''
            probs, hids, mems = self.decoder(h_t, *mems)

            probs = probs.masked_fill(mask, 0)

            if self.deterministic:
                city = probs.argmax(dim=-1)  # (1, B)
            else:
                city = Categorical(probs).sample()  # (1, B)
            
            city_idx = city.squeeze()  # (B)

            tour.append(city)
            probs_cat.append(probs)

            chosen_prob = probs[:, toB, city_idx]  # (1, B)
            log_probs.append(chosen_prob.log())
            
            # update mask 
            if mask is not None:
                mask = mask.clone()  # solution?
                mask[:, toB, city_idx] = True
            
            # update mems
            mems = self._update_mems(hids, mems, segment_len)

            h_t = h_enc[city_idx, toB, :].unsqueeze(0)  # (1, B, H)

            # compute loss ? (for SL)

            # Loss
            # target_t = target[:,t]  # (B, 1)
            # loss = self.criterion(city, target_t)
            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()
        
        tour = torch.cat(tour, dim=0)  # (N, B)
        sum_log_probs = torch.cat(log_probs, dim=0).sum(dim=0)  # (B)
        probs_cat = torch.cat(probs_cat, dim=0)  # (N, B, N)
            
        return tour, sum_log_probs, probs_cat, mems  #, loss

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # device = torch.device('cuda')
    device = torch.device('cpu')

    bsz, d_model, n_class = 100, 128, 25
    crit = nn.NLLLoss()
    oz = torch.optim.Adam
    model = TSPXL(rl=True, d_model=d_model, d_ff=512, n_head=8, n_enc_layer=6, n_dec_layer=2, bsz=bsz, deterministic=False, criterion=crit,
                  dropout_rate=0.1, internal_drop=-1, clip_value=-1, pre_lnorm=True, clamp_len=-1).to(device)

    mems = tuple()

    x = torch.randn(n_class, bsz, 2).to(device)
    targ = torch.randint(0,n_class,(bsz,n_class)).to(device)

    ret = model(x, targ, *mems)
    tour, log_probs, probs_cat, new_mems = ret

    ret2 = model(x, targ, *new_mems)
    print('tour=', tour.shape)
    print(tour)
    print('new_mems=',len(new_mems), new_mems[0].shape)