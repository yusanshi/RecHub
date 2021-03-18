import torch
import dgl

from .lightgcn_conv import LightGCNConv
from ..base import HeterogeneousAggregator


def get_index(a, b):
    '''
    For each value in b, get its index in a and return the indexs
    Args:
        a: 1D tensor
        b: 1D tensor
    Returns:
        indexs (the same shape as b)
    '''
    assert len(a.size()) == 1
    assert len(b.size()) == 1
    index = a.argsort()
    return index[torch.bucketize(b, a[index])]


class LightGCN(HeterogeneousAggregator):
    def __init__(self, graph_embedding_dims, etypes):
        super().__init__(graph_embedding_dims, etypes)

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        assert 0 <= current_layer <= total_layer - 1

        return LightGCNConv()

    # TODO
    # I found that adding this (i.e., mean over all layer outputs) contributes to a lower score.
    # Maybe there is something wrong with my implementations.
    # Anyway, I commented out this to make its default behavior take effect (i.e. use only the last layer output)
    # for a better performance

    # def single_forward(self, layers, blocks, h):
    #     outputs = []
    #     for layer, block in zip(layers, blocks):
    #         if len(block.srcdata[dgl.NID]) == 0:
    #             # If is not a block.
    #             outputs.append(h)
    #         else:
    #             # Since blocks have different size of input and output nodes,
    #             # We should only save the nodes available in the last block output
    #             outputs.append({
    #                 k: v[get_index(block.srcdata[dgl.NID][k],
    #                                blocks[-1].dstdata[dgl.NID][k])]
    #                 for k, v in h.items()
    #             })
    #         h = layer(block, h)
    #     outputs.append(h)
    #     outputs = {
    #         k: torch.mean(torch.stack([x[k] for x in outputs], dim=-1), dim=-1)
    #         for k in outputs[0].keys()
    #     }
    #     return outputs
