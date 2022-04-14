#!/usr/bin/env python3
from email.policy import default
import os.path as osp
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch.nn import Linear

from dataset import *
from torch_sparse import SparseTensor

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, default="./dataset/npz/", help="the directory path to graphs")
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=1000, help="number of epoches")
parser.add_argument("--model", type=str, default='gcn', choices=['gcn', 'gin'], help="type of model")
parser.add_argument("--training", type=str, default='training', choices=['training', 'inference'], help="training or inference")
parser.add_argument("--sparseTensor", type=int, default=0, help="sparseTensor")

args = parser.parse_args()
print(args)

path = osp.join(args.dataDir, args.dataset+".npz")
# path = osp.join("/home/yuke/.graphs/orig/", args.dataset)
dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=False)
data = dataset


adj = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])

if(args.sparseTensor == 1):
    adj_tensor = adj
else:
    adj_tensor = data.edge_index 

# normalization
g = data.g
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm = norm.cuda()
data.edge_attr = norm


if args.model == 'gcn':
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden, cached=False,
                                normalize=False)
            self.conv2 = GCNConv(args.hidden, dataset.num_classes, cached=False,
                                normalize=False)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

            x = F.relu(self.conv1(x, adj_tensor, edge_weight))
            x = self.conv2(x, adj_tensor, edge_weight)

            return x
else:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            num_features = dataset.num_features
            dim = 64

            input_fc =  Linear(num_features, dim)
            hidden_fc = Linear(dim, dim)
            output_fc = Linear(dim, dataset.num_classes)

            self.conv1 = GINConv(input_fc)
            self.conv2 = GINConv(hidden_fc)
            self.conv3 = GINConv(hidden_fc)
            self.conv4 = GINConv(hidden_fc)
            self.conv5 = GINConv(output_fc)

        def forward(self):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, adj_tensor)
            x = self.conv2(x, adj_tensor)
            x = self.conv3(x, adj_tensor)
            x = self.conv4(x, adj_tensor)
            x = self.conv5(x, adj_tensor)
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

    if args.model == 'gcn':
        print("GCN (L2-H16) Train -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))
    else:
        print("GIN (L5-H64) Train -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))
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

    if args.model == 'gcn':
        print("GCN (L2-H16) Inference -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))
    else:
        print("GIN (L5-H64) Inference -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))
    print()


