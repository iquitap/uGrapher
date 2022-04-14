#!/usr/bin/env python3
import os.path as osp
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch.nn import Linear

from dataset import *
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops


parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, default="./dataset/npz/", help="the directory path to graphs")
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=1000, help="number of epoches")
parser.add_argument("--model", type=str, default='gat', choices=['gat'], help="type of model")
parser.add_argument("--training", type=str, default='training', choices=['training', 'inference'], help="training or inference")

parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")

parser.add_argument("--sparseTensor", type=int, default=0, help="sparseTensor")


args = parser.parse_args()
print(args)

path = osp.join(args.dataDir, args.dataset+".npz")
# path = osp.join("/home/yuke/.graphs/orig/", args.dataset)
dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=False)
data = dataset

data.edge_index, _ = add_self_loops(data.edge_index)

adj = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])

if(args.sparseTensor == 1):
    adj_tensor = adj
else:
    adj_tensor = data.edge_index 


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GATConv(dataset.num_features, args.hidden, heads=args.num_heads, dropout=0.0, bias=False, add_self_loops = False)
        self.conv2 = GATConv(args.hidden * args.num_heads, dataset.num_classes, heads=1, concat=False,
                             dropout=0.0, bias=False, add_self_loops = False)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, adj_tensor))
        x = self.conv2(x, adj_tensor)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


if args.training == 'training':
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in tqdm(range(1, args.epochs + 1)):
        train()
    torch.cuda.synchronize()
    dur = time.perf_counter() - start

    print("GAT (L2-H8) Train -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))
    print()
else: # inference
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in tqdm(range(1, args.epochs + 1)):
        # train()
        model.eval()
        logist = model()
    torch.cuda.synchronize()
    dur = time.perf_counter() - start

    print("GAT (L2-H8) Inference -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))

    print()


