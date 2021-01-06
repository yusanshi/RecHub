import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(in_feats, hidden_size)
        self.layer2 = GraphConv(hidden_size, out_feats)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = torch.relu(h)
        h = self.layer2(g, h)
        return h
