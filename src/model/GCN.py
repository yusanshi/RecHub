import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

# TODO add dropout?


class GCN(nn.Module):
    def __init__(self, graph_embedding_dims):
        super(GCN, self).__init__()
        assert len(graph_embedding_dims) >= 2
        self.layers = nn.ModuleList()
        for i in range(len(graph_embedding_dims) - 2):
            self.layers.append(
                GraphConv(graph_embedding_dims[i],
                          graph_embedding_dims[i + 1],
                          activation=nn.ReLU()))
        self.layers.append(
            GraphConv(graph_embedding_dims[-2], graph_embedding_dims[-1]))

    def forward(self, g, features):
        g = dgl.add_self_loop(g)
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


if __name__ == '__main__':
    import torch
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = GCN([16, 12, 8, 6]).to(device)
    print(model)
    inputs = torch.rand(10, 16).to(device)
    print(model(graph, inputs))
