import torch
from torch import nn

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, internal_drop, clip):
        """
        internal_drop : Apply dropout after the calculating attention scores. Expected input: dropout ratio, disable if -1
        clip : clip value, disable if -1
        """
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.scale = 1 / (self.d_head ** 0.5)
        self.clip = clip
        if internal_drop > 0:
            self.drop = nn.Dropout(internal_drop)
    
    def _rel_shift(self, x):
        zero_pad = torch.zeros(x.size(0), 1, *x.size()[2:], device=x.device, dtype=x.dtype)
        x_pad = torch.cat([zero_pad, x], dim=1)
        x_pad = x_pad.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_pad[1:].view_as(x)
        return x

    def forward(self):
        raise NotImplementedError
        
class MultiHeadSelfAttn(MultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def forward(self, q, k, v):
        """
        Notes:
            nh : n_head
            D : H/nh = d_model / n_head
        Inputs:
            q : query token (1, B, H)
            k : key vectors (N, B, H)
            v : value vectors (N, B, H)
            All inputs are weighted with Wq, Wk and Wv
        Outputs:
            out : 
        """
        qlen, bsz, klen = q.size(0), q.size(1), k.size(0)

        q = q.view(qlen, bsz, self.n_head, self.d_head)  # (1, B, nh, D)
        k = k.view(klen, bsz, self.n_head, self.d_head)  # (N, B, nh, D)
        v = v.view(klen, bsz, self.n_head, self.d_head)  # (N, B, nh, D)

        score = torch.einsum('ibnd,jbnd->ijbn', (q, k))  # query(1, B, nh, D) x key(N, B, nh, D) => score(1, N, B, nh)
        score.mul_(self.scale)

        if self.clip > 0:
            score = self.clip * torch.tanh(score)

        # Masking

        weight = torch.softmax(score, dim=1)
        try: weight = self.drop(weight)
        except: pass

        out = torch.einsum('ijbn,jbnd->ibnd', (weight, v))  # weight(1, N, B, nh) x value(N, B, nh, D) => out(1, B, nh, D)

        out = out.view(qlen, bsz, self.n_head * self.d_head).contiguous()  # (1, B, H)

        return out

class RelMultiHeadAttn(MultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def forward(self, q, k, v, r, bias_u, bias_v, mask):
        """
        q : query token (1, B, H)
        k : key vector (2N or N, B, H)
        v : value vector (2N or N, B, H)
        r : rel PE (N, 1, H)
        bias : u, v in Dai et al. 2019 (nh, D)
        All inputs are weighted with Wq, Wk, Wv and Wr
        
        """
        qlen, bsz, klen, rlen = q.size(0), q.size(1), k.size(0), r.size(0)

        q = q.view(qlen, bsz, self.n_head, self.d_head)  # (1, B, nh, D)
        k = k.view(klen, bsz, self.n_head, self.d_head)  # (N, B, nh, D)
        v = v.view(klen, bsz, self.n_head, self.d_head)  # (N, B, nh, D)
        r = r.view(rlen, self.n_head, self.d_head)  # (N, nh, D)

        
        # AC = (W_qE+u)W_kE
        qu = q + bias_u  # W_qE + u
        AC = torch.einsum('ibnd,jbnd->ijbn', (qu, k))  # qu(1, B, nh, D) x k(N, B, nh, D) => (1, N, B, nh)
        # BD = (W_qE+v)W_kR
        qv = q + bias_v # W_qE + v
        BD = torch.einsum('ibnd,jnd->ijbn', (qv, r))  # qv(1, B, nh, D) x r(N, nh, D) => (1, N, B, nh)
        BD = self._rel_shift(BD)

        score = AC + BD  # (1, N, B, nh)
        score.mul_(self.scale)

        if self.clip > 0:
            score = self.clip * torch.tanh(score)

        weight = torch.softmax(score, dim=1)  # (1, N, B, nh)
        try: weight = self.drop(weight)
        except: pass

        # Calculate attention probs
        if mask is not None:
            logits = score.mean(dim=-1)  # (1, N, B)
            logits.masked_fill_(mask, -torch.inf)
            probs = torch.softmax(logits, dim=1)  # (1, N, B)
        else:
            probs = weight.mean(dim=-1)


        out = torch.einsum('ijbn,jbnd->ibnd', (weight, v))
        out = out.view(qlen, bsz, self.n_head * self.d_head).contiguous()  # (1, B, H)

        return out, probs