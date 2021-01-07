import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model.NGCF.NGCFConv import NGCFConv


class NGCF(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(NGCF, self).__init__()
        self.layer1 = NGCFConv(in_feats, hidden_size)
        self.layer2 = NGCFConv(hidden_size, out_feats)

    def forward(self, g, inputs):
        g = dgl.remove_self_loop(g)
        h1 = self.layer1(g, inputs)
        h1 = F.leaky_relu(h1)
        h2 = self.layer2(g, h1)
        h2 = F.leaky_relu(h2)
        h3 = self.layer2(g, h2)
        h = torch.cat((h1, h2, h3), dim=-1)
        return h


if __name__ == '__main__':
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = NGCF(16, 16, 16).to(device)
    inputs = torch.rand(10, 16).to(device)
    print(model(graph, inputs))
