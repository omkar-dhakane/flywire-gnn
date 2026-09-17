"""Baseline models for FlyWire FAFB node classification."""

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=None, num_layers=3, dropout=0.5):
        super().__init__()
        out_dim = out_dim or hidden
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for layer in self.layers[:-1]:
            x = F.dropout(F.relu(layer(x)), self.dropout, training=self.training)
        return self.layers[-1](x)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=None, num_layers=3, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv

        out_dim = out_dim or hidden
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=None, num_layers=3, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        out_dim = out_dim or hidden
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
