import torch
import torch.nn as nn

import dgl.function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair


def has_self_loop(graph):
    return any(graph.edges()[0] - graph.edges()[1] == 0)


class NGCFConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(NGCFConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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
            # TODO norm for interaction part?

            graph.srcdata['h_self'] = feature
            graph.update_all(fn.copy_u('h_self', 'm_self'),
                             fn.sum('m_self', 'h_self'))

            def message_func(edges):
                return {
                    'm_interaction':
                    torch.mul(edges.src['h_interaction'],
                              edges.dst['h_interaction'])
                }

            graph.srcdata['h_interaction'] = feature
            graph.update_all(message_func,
                             fn.sum('m_interaction', 'h_interaction'))

            rst = torch.matmul(
                graph.dstdata['h_self'], self.weight_self) + torch.matmul(
                    graph.dstdata['h_interaction'], self.weight_interaction)

            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1, ) * (feature.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            rst = (rst + feature_original) / 2  # TODO

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        return f'in={self.in_features}, out={self.out_features}'
