import torch
from torch import nn

class TSPEncoder(nn.Module):
    """
    Encodes 2-D Euclidean TSP instance into context tensor by applying MHSA
    Notes:
        N : number of points in current segment (=L_seg)
        B : batch size
        H : hidden dimension
    """
    def __init__(self, d_model, d_ff, n_head, n_layer, dropout_rate=0.2, device='cpu'):
        ''' docstring '''
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.device = device

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
            h, attn_weights = self.MHA_layers[l](h, h, h)
            
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

            return h, attn_weights

if __name__ == '__main__':
    model = TSPEncoder(d_model=128, d_ff=512, n_head=8, n_layer=8)
    N, B, H = 50, 100, 128
    instance = torch.rand(N+1, B, H)
    out, weight = model(instance)
    print('Out=', out.shape)
    print('Weight=', weight.shape)