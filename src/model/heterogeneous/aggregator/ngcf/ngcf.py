import torch.nn as nn
import dgl
from .ngcf_conv import NGCFConv


class NGCF(nn.Module):
    def __init__(self, graph_embedding_dims):
        super(NGCF, self).__init__()
        assert len(graph_embedding_dims) >= 2
        self.layers = nn.ModuleList()
        for i in range(len(graph_embedding_dims) - 2):
            self.layers.append(
                NGCFConv(graph_embedding_dims[i],
                         graph_embedding_dims[i + 1],
                         activation=nn.LeakyReLU()))
        self.layers.append(
            NGCFConv(graph_embedding_dims[-2], graph_embedding_dims[-1]))

    def forward(self, g, features):
        g = dgl.remove_self_loop(g)
        outputs = []
        h = features
        outputs.append(h)
        for layer in self.layers:
            h = layer(g, h)
            outputs.append(h)
        return torch.cat(outputs, dim=-1)


if __name__ == '__main__':
    import torch
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = NGCF([16, 12, 8, 6]).to(device)
    print(model)
    inputs = torch.rand(10, 16).to(device)
    print(model(graph, inputs))
