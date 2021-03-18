import torch.nn as nn
import dgl

from .ngcf_conv import NGCFConv
from ..base import HeterogeneousAggregator


class NGCF(HeterogeneousAggregator):
    def __init__(self, graph_embedding_dims, etypes):
        super().__init__(graph_embedding_dims, etypes)

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        assert 0 <= current_layer <= total_layer - 1
        if current_layer < total_layer - 1:
            return NGCFConv(input_dim, output_dim, activation=nn.LeakyReLU())

        return NGCFConv(input_dim, output_dim)

    # TODO NGCF should concat the outputs of all layers
    # def single_forward(self, layers, blocks, h):
    #     outputs = []
    #     outputs.append(h)
    #     for layer, block in zip(layers, blocks):
    #         h = layer(block, h)
    #         outputs.append(h)
    #     return torch.cat(outputs, dim=-1)


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
