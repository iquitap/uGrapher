import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn import GATConv
# from model.gat_06 import GATConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=True))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, allow_zero_in_degree=True))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 aggregator_type):
        super(GraphSAGE, self).__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, inputs):
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h
class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP"""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h


class GIN(nn.Module):
    """GIN model"""

    def __init__(self, graph, input_dim, hidden_dim, output_dim, num_layers=5):
        """model parameters setting
        Paramters
        ---------
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(GIN, self).__init__()
        self.g = graph
        self.num_layers = num_layers
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                mlp = nn.Linear(input_dim, hidden_dim)
            elif layer < self.num_layers - 1:
                mlp = nn.Linear(hidden_dim, hidden_dim)
            else:
                mlp = nn.Linear(hidden_dim, output_dim)

            self.ginlayers.append(GINConv(ApplyNodeFunc(mlp), "sum", init_eps=0, learn_eps=False))

    def forward(self, h):
        for i in range(self.num_layers):
            h = self.ginlayers[i](self.g, h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()

        assert n_layers >= 2
        self.layers.append(GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))

        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))

        self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        # return F.log_softmax(h, dim=1)
        return h
