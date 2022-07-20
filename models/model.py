import torch
from torch import nn
from torch.distributions import Categorical

from encoder import TSPEncoder
from decoder import TSPDecoder

class TSPXL(nn.Module):
    """
    Encoder:
        H_enc = Encoder(x)  # Encodes segmented TSP instance into context vectors with MHSA and FFN layers
    Decoder:
        Decodes autoregressively for L_seg tokens
        e.g.)
        For first segment, do MHA until the last token. And store H_enc to memory
        For first token of second segment, do MHA with mems[1:] + current H_enc[:1] as Keys and Values 
        For second token of second segment, do MHA with mems[2:] + current H_enc[:2] as Keys and Values
        For third token of second segment, do MHA with mems[3:] + current H_enc[:1] as Keys and Values
        and so on ...


    """
    def __init__(self, d_model, d_ff, n_head, n_layer, n_class, bsz, deterministic, criterion, optimizer,
                       dropout_rate, internal_drop, clip_value, pre_lnorm, clamp_len):
        super().__init__()
        self.d_model = d_model
        self.n_class = n_class
        self.n_layer = n_layer
        self.deterministic = deterministic


        self.input_emb = nn.Linear(2, d_model)
        self.start_tokens = nn.Parameter(torch.randn(d_model))
        self.encoder = TSPEncoder(d_model, d_ff, n_head, n_layer)
        self.decoder = TSPDecoder(d_model, d_ff, n_layer, n_head, n_class, clamp_len,
                                  bsz, dropout_rate, internal_drop, clip_value, pre_lnorm)

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters())
    
    def _init_mems(self):
        mems = []
        param = next(self.parameters())
        for i in range(self.n_layer + 1):
            empty = torch.empty(0, dtype=param.dtype, device=param.device)
            mems.append(empty)
        return mems

    def _update_mems(self, hids, mems, segment_len, first=False):
        if mems is None: return None

        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        if first:
            pass
        else:
            with torch.no_grad():
                new_mems = []
                start_idx = segment_len
                for i in range(len(hids)):
                    cat = torch.cat([mems[i], hids[i]], dim=0)  # (N+1, B, H)
                    new_mems.append(cat[start_idx:].detach())
        return new_mems

    def forward(self, x, target, mask, *mems):
        """
        Inputs
            x : 2D Euclidean TSP instance (segmented data)  # size=(N, B, 2)
            target :   # size=(B, N)
            mask : Initially zeros(B, H, nc), keep being updated in `for t loop`
            mems : 
                Initially tuple()
                If this segment is the first one, mems is always H_enc
                Otherwise, mems will be updated along `for t loop`, maintaining segment length
        Outputs
            loss : 
            # tour : Sequence of city indices including start token  # size=()
            # sum_log_probs : Sum of log probabilities of action (city choice)  # size=()
        """
        segment_len = x.size(0)
        bsz = x.size(1)
        toB = torch.arange(bsz)
        
        h = self.input_emb(x)  # (N, B, H)

        # Concat with start token
        # h = torch.cat([h, self.start_tokens.repeat(bsz, 1, 1)], dim=0)  # (N+1, B, H)

        # Encode it !
        h_enc, _ = self.encoder(h)  # (N, B, H)

        tour = []
        sum_log_probs = []
        # Decode it !
        for t in range(segment_len):
            h_t = h_enc[t:t+1]

            if not mems: mems = self._init_mems()
            '''
            probs : (B, 1, nc)
            hids, mems : len=9, size(hid)=(1,B,H), size(mem)=(N,B,H)
            '''
            probs, hids, mems = self.decoder(h_t, mask, mems=mems) 

            if self.deterministic:
                city = probs.argmax(dim=-1)  # (B, 1)
            else:
                city = Categorical(probs).sample()  # (B, 1)

            tour.append(city)
            
            # chosen_prob = probs[toB, city.squeeze_()]  # (B) for REINFORCE
            # sum_log_probs.append(chosen_prob.log())  # for REINFORCE
            
            # update mask 
            mask_temp = torch.zeros(bsz, self.n_class)  # (B, nc)
            mask_temp[torch.arange(bsz), city.squeeze()] = 1
            mask_temp = mask_temp[:,None,:].repeat(1, self.d_model, 1)  # (B, H, nc)
            mask = mask + mask_temp
            
            # update mems
            new_mems = self._update_mems(hids, mems, segment_len)

            # Loss
            target_t = target[:,t]  # (B, 1)
            loss = self.criterion(city, target_t)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        tour = torch.cat(tour, dim=1)
            
        return tour, loss, new_mems

if __name__ == '__main__':
    bsz, d_model, n_class = 100, 128, 50
    crit = nn.NLLLoss()
    oz = torch.optim.Adam
    model = TSPXL(d_model=d_model, d_ff=512, n_head=8, n_layer=8, n_class=n_class, bsz=bsz, deterministic=False, criterion=crit, optimizer=oz,
                  dropout_rate=0.1, internal_drop=-1, clip_value=-1, pre_lnorm=True, clamp_len=-1)

    mask = torch.zeros(bsz, d_model, n_class)
    mems = tuple()

    x = torch.randn(n_class, bsz, 2)
    targ = torch.randint(0,n_class,(bsz,n_class))

    ret = model(x, targ, mask, *mems)
    tour, loss, new_mems = ret
    print('tour=', tour.shape)
    print(tour)
    print('loss=', loss.shape)
    print('new_mems=',len(new_mems), new_mems[0].shape)