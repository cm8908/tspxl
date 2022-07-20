import torch

def generate_APE(self):
    '''
    Generate absolute positional encoding (Code from Bresson et al. 2021)
    '''
    ape = torch.zeros(self.max_len, self.d_model, device=self.device)
    pos = torch.arange(0, self.max_len, device=self.device)
    div = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
    ape[:,0::2] = torch.sin(pos * div)
    ape[:,1::2] = torch.cos(pos * div)
    return ape