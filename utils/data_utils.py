import os
import torch
import numpy as np
from torch.utils.data import Dataset

class RandomTSPGenerator:
    def __init__(self, bsz, total_len, max_step, device='cpu', segm_len=-1):
        self.bsz = bsz
        self.max_step = max_step
        self.total_len = total_len
        self.segm_len = segm_len

        self.device = device
    
    def make_batch(self):
        batch = torch.rand(self.total_len, self.bsz, 2).to(self.device)  # (N, B, 2)
        return batch

    def get_split_iter(self, batch):
            n_segm = batch.size(1) // self.segm_len
            for j in range(0, self.total_len, self.segm_len):
                done = j == self.total_len // self.segm_len - 1
                try:
                    yield batch[j:j+self.segm_len,:,:], done
                except IndexError:
                    yield batch[j:j+self.segm_len,:,:], True
                

    def get_fixlen_iter(self):
        for i in range(self.max_step):
            yield self.make_batch()

    def get_varlen_iter(self):
        raise NotImplementedError

    def __iter__(self):
        return self.get_fixlen_iter()

class SortedTSPGenerator(RandomTSPGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def sort(self, x):
        '''x: (N, B, 2) '''
        xn = x.numpy()
        xy = xn[:,:,0] + xn[:,:,1]  # (N, B)
        idx_origin = np.arange(x.shape[0])
        xy_srt =  np.sort(xy, axis=0)[::-1]  # (N, B)
        res = []
        for b in range(self.bsz):
            mapper = {xy_i: idx for xy_i, idx in zip(xy[:,b], idx_origin)}
            idx_srt = [mapper[xy_i] for xy_i in xy_srt[:,b]]
            xn_srt = np.array([xn[i,b,:] for i in idx_srt])
            res.append(xn_srt)
        res = np.stack(res, axis=1)  # (N, B, 2)
        return torch.Tensor(res)

    def get_sorted_iter(self):
        for i in range(self.max_step):
            batch = self.make_batch()
            batch_sorted = self.sort(batch)
            yield batch_sorted

        pass
    def __iter__(self):
        return self.get_sorted_iter()


class TSPDataset(Dataset):
    """
    B = total length of dataset
    self.data : total records of all n (x,y) samples (Tensor shape B x N x 2)
    self.label : concorde solutions of tsp tour index sequence (Tensor shape B x N)
    Return: data, label indices  #, distance matrix, modified adjacency matrix, adjacency matrix, label tour length
    """
    # TODO: implement split_iter
    def __init__(self, n, mode, root_dir='datasets', author='joshi', device='cpu', segm_len=-1):

        filename = f'tsp{n}_{mode}_concorde.txt'
        filename = os.path.join(root_dir, author+'-data', filename)
        self.n = n
        self.data = []
        self.label = []
        with open(filename, 'r') as file:
            for line in file:
                if line == '\n':
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
    # trainset = TSPDataset(n=10, mode='train', root_dir='/home/gailab/ms/tspxl/datasets', author='joshi', device='cpu')
    # valset = TSPDataset(n=10, mode='val', root_dir='/home/gailab/ms/tspxl/datasets', author='joshi', device='cpu')
    # testset = TSPDataset(n=10, mode='test', root_dir='/home/gailab/ms/tspxl/datasets', author='joshi', device='cpu')
    # print(f'train: {len(trainset)}, val: {len(valset)}, test: {len(testset)}')

    gen = SortedTSPGenerator(bsz=2, total_len=10, segm_len=5, max_step=1, device='cpu')
    for i, batch in enumerate(gen):
        for split_batch, done in gen.get_split_iter(batch):
            print(split_batch.shape, done)
            if i == 0:
                print(split_batch[:,0,:], split_batch[:,0,0]+split_batch[:,0,1])
    # data, label = dataset[0]
    # print(len(dataset))
    # print(data.shape, label.shape)
    # print(label)