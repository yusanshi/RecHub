import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    # TODO how to handle heads if use `graph_embedding_dims`
    def __init__(self, in_feats, hidden_size, out_feats, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_feats, hidden_size, num_heads)
        self.layer2 = GATConv(hidden_size * num_heads, out_feats, 1)

    def forward(self, g, inputs):
        g = dgl.add_self_loop(g)
        h = self.layer1(g, inputs)
        h = h.view(h.size(0), -1)
        h = F.elu(h)
        h = self.layer2(g, h).squeeze(dim=1)
        return h


if __name__ == '__main__':
    import torch
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = GAT(16, 4, 16, 4).to(device)
    inputs = torch.rand(10, 16).to(device)
    print(model(graph, inputs))
