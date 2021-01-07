import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(in_feats, hidden_size)
        self.layer2 = GraphConv(hidden_size, out_feats)

    def forward(self, g, inputs):
        g = dgl.add_self_loop(g)
        h = self.layer1(g, inputs)
        h = F.relu(h)
        h = self.layer2(g, h)
        return h


if __name__ == '__main__':
    import torch
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = GCN(16, 16, 16).to(device)
    inputs = torch.rand(10, 16).to(device)
    print(model(graph, inputs))
