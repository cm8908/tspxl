import os
import torch
from torch.utils.data import Dataset

class RandomTSPGenerator:
    def __init__(self, bsz, seq_len, max_step, device='cpu', ext_len=None):
        self.bsz = bsz
        self.max_step = max_step
        self.seq_len = seq_len  # tgt_len
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
    
    def get_batch(self):
        data = torch.rand(self.seq_len, self.bsz, 2).to(self.device)  # (N, B, 2)
        return data

    def get_fixlen_iter(self):
        for i in range(self.max_step):
            yield self.get_batch()

    def get_varlen_iter(self):
        raise NotImplementedError

    def __iter__(self):
        return self.get_fixlen_iter()

class TSPDataset(Dataset):
    pass

class TSPDataset(Dataset):
    """
    B = total length of dataset
    self.data : total records of all n (x,y) samples (Tensor shape B x N x 2)
    self.label : concorde solutions of tsp tour index sequence (Tensor shape B x N)
    Return: data, label indices  #, distance matrix, modified adjacency matrix, adjacency matrix, label tour length
    """
    def __init__(self, n, mode, root_dir='datasets', author='joshi', device='cpu'):

        filename = f'tsp{n}_{mode}_concorde.txt'
        filename = os.path.join(root_dir, author+'-data', filename)
        self.n = n
        self.data = []
        self.label = []
        with open(filename, 'r') as file:
            for line in file:
                if line is '\n':
                    break
                sample = line.split(' ')    
                xs = list(map(float, sample[:2*n-1:2]))
                ys = list(map(float, sample[1:2*n:2]))
                sample_data = [xs, ys]
                sample_label = list(map(int, sample[2*n+1:-1]))

                self.data.append(sample_data)
                self.label.append(sample_label)
        if len(self.data) != len(self.label):
            raise ValueError(f'length of data {len(self.data)} while label {len(self.label)}')

        # B x 2 x N
        self.data = torch.Tensor(self.data).transpose(1,2)
        self.label = torch.LongTensor(self.label)

        # self.dist, self.adj, self.real_adj, self.tour_len = self.make_matrices(k)

        self.data = self.data.to(device)
        self.label = self.label.to(device)
        # self.dist = self.dist.to(device)
        # self.adj = self.adj.to(device)
        # self.real_adj = self.real_adj.to(device)
        # self.tour_len = self.tour_len.to(device)
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx] #, self.dist[idx], self.adj[idx].long(), self.real_adj[idx].long(), self.tour_len[idx]

if __name__ == '__main__':
    dataset = TSPDataset(n=10, mode='train', root_dir='/home/gailab/ms/tspxl/datasets', author='joshi', device='cpu')
    data, label = dataset[0]
    print(len(dataset))
    print(data.shape, label.shape)
    print(label)