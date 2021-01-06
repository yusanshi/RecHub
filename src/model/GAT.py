import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_feats, hidden_size, num_heads)
        self.layer2 = GATConv(hidden_size * num_heads, out_feats, 1)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = h.view(h.size(0), -1)
        h = F.elu(h)
        h = self.layer2(g, h).squeeze(dim=1)
        return h
