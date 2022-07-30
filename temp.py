import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)
log_probs = []
layer = nn.Linear(28, 28)
linear = nn.Linear(28, 5)
h_t = torch.randn(1, 10, 28)
mask = torch.zeros(1, 10, 5).bool()
for i in range(5):
    h_t = layer(h_t)
    prob = linear(h_t)
    prob = prob.masked_fill(mask, 0)
    city = prob.argmax(-1)  # Cat?
    mask[:,torch.arange(10),city] = True
log_probs.append(prob.log())
L_diff = 10
slp = torch.cat(log_probs, dim=0).sum(dim=0)
loss = L_diff * slp
loss.mean().backward()