import torch
import torch.nn as nn

import dgl.function as fn
from dgl.base import DGLError


def has_self_loop(graph):
    return any(graph.edges()[0] - graph.edges()[1] == 0)


class NGCFConv(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight_self = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_interaction = nn.Parameter(
            torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight_self)
        nn.init.xavier_uniform_(self.weight_interaction)

    def forward(self, graph, feature):
        with graph.local_scope():
            # TODO very slowly
            # if has_self_loop(graph):
            #     raise DGLError(
            #         'You should first remove self loop with `dgl.remove_self_loop`.'
            #     )

            feature_original = feature

            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1, ) * (feature.dim() - 1)
            norm = torch.reshape(norm, shp)
            feature = feature * norm

            graph.srcdata['h'] = feature
            graph.srcdata['h_original'] = feature_original

            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_self'))

            def message_func(edges):
                return {
                    'm': torch.mul(edges.src['h'], edges.dst['h_original'])
                }

            graph.update_all(message_func, fn.sum('m', 'h_interaction'))

            rst = torch.matmul(
                graph.dstdata['h_self'], self.weight_self) + torch.matmul(
                    graph.dstdata['h_interaction'], self.weight_interaction)

            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1, ) * (feature.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            rst = rst + torch.matmul(feature_original, self.weight_self)

            if self.activation is not None:
                rst = self.activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        return f'in={self.in_features}, out={self.out_features}, activation={self.activation}'


if __name__ == '__main__':
    import dgl
    graph = dgl.graph(([0, 0, 1, 2, 2], [1, 2, 2, 1, 0]))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = NGCFConv(2, 2).to(device)
    inputs = torch.tensor([
        [0.2, 0.5],
        [0.6, 0.4],
        [0.3, 0.3],
    ]).to(device)
    print(model(graph, inputs))
