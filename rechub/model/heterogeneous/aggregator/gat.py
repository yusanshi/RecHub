import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv

from .base import HeterogeneousAggregator


class GAT(HeterogeneousAggregator):
    def __init__(self, graph_embedding_dims, etypes, num_attention_heads):
        self.num_attention_heads = num_attention_heads  # Must put this before super().__init__()
        super().__init__(graph_embedding_dims, etypes)

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        assert 0 <= current_layer <= total_layer - 1
        if current_layer < total_layer - 1:
            return GATConv(
                input_dim *
                (self.num_attention_heads if current_layer >= 1 else 1),
                output_dim,
                self.num_attention_heads,
                activation=nn.ELU())

        return GATConv(
            input_dim * (self.num_attention_heads if total_layer >= 2 else 1),
            output_dim, self.num_attention_heads)

    def single_forward(self, layers, blocks, h):
        for layer, block in zip(layers, blocks):
            h = layer(block, h)
            h.view(h.size(0), -1)
        return h


if __name__ == '__main__':
    import torch
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = GAT([8, 6], 8).to(device)
    inputs = torch.rand(10, 8).to(device)
    print(model(graph, inputs))
    model = GAT([16, 12, 8, 6], 8).to(device)
    inputs = torch.rand(10, 16).to(device)
    print(model(graph, inputs))
