import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self, graph_embedding_dims, num_attention_heads,
                 heads_aggregating_method):
        super(GAT, self).__init__()
        assert len(graph_embedding_dims) >= 2
        assert heads_aggregating_method in ['avg', 'cat']
        self.heads_aggregating_method = heads_aggregating_method
        self.layers = nn.ModuleList()
        for i in range(len(graph_embedding_dims) - 2):
            self.layers.append(
                GATConv(graph_embedding_dims[i] *
                        (num_attention_heads if
                         heads_aggregating_method == 'cat' and i >= 1 else 1),
                        graph_embedding_dims[i + 1],
                        num_attention_heads,
                        activation=nn.ELU()))
        self.layers.append(
            GATConv(
                graph_embedding_dims[-2] *
                (num_attention_heads if heads_aggregating_method == 'cat'
                 and len(graph_embedding_dims) >= 3 else 1),
                graph_embedding_dims[-1], num_attention_heads))

    def forward(self, g, features):
        g = dgl.add_self_loop(g)
        h = features
        for layer in self.layers:
            h = layer(g, h)
            if self.heads_aggregating_method == 'avg':
                h = h.mean(dim=1)
            elif self.heads_aggregating_method == 'cat':
                h = h.view(h.size(0), -1)
            else:
                raise NotImplementedError
        return h


if __name__ == '__main__':
    import torch
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    for x in ['avg', 'cat']:
        model = GAT([8, 6], 8, x).to(device)
        inputs = torch.rand(10, 8).to(device)
        print(model(graph, inputs))
        model = GAT([16, 12, 8, 6], 8, x).to(device)
        inputs = torch.rand(10, 16).to(device)
        print(model(graph, inputs))
