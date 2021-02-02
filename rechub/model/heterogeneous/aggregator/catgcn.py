import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

from .base import HeterogeneousAggregator


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


# Two slow
# def get_index(a, b):
#     '''
#     For each value in b, get its index in a and return the indexs
#     Args:
#         a: 1D tensor
#         b: 1D tensor
#     Returns:
#         indexs (the same shape as b)
#     '''
#     assert len(a.size()) == 1
#     assert len(b.size()) == 1
#     return torch.nonzero(
#         a.expand(b.size(0), -1) == b.expand(a.size(0), -1).transpose(0, 1))[:,
#                                                                             1]


class CATGCN(HeterogeneousAggregator):
    def __init__(self, graph_embedding_dims, etypes):
        super().__init__(graph_embedding_dims, etypes)

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        assert 0 <= current_layer <= total_layer - 1
        if current_layer < total_layer - 1:
            return GraphConv(input_dim, output_dim, activation=nn.ReLU())

        return GraphConv(input_dim, output_dim)

    def single_forward(self, layers, blocks, h):
        outputs = []
        for layer, block in zip(layers, blocks):
            if len(block.srcdata[dgl.NID]) == 0:
                # If is not a block. TODO: better if condidation?
                outputs.append(h)
            else:
                # Since blocks have different size of input and output nodes,
                # We should only save the nodes available in the last block output
                outputs.append({
                    k: v[get_index(block.srcdata[dgl.NID][k],
                                   blocks[-1].dstdata[dgl.NID][k])]
                    for k, v in h.items()
                })
            h = layer(block, h)
        outputs.append(h)
        outputs = {
            k: torch.cat([x[k] for x in outputs], dim=-1)
            for k in outputs[0].keys()
        }
        return outputs


if __name__ == '__main__':
    import torch
    from ....utils import add_reverse
    graph = dgl.heterograph(
        add_reverse({
            ('user', 'follow', 'category'):
            (torch.randint(10, (100, )), torch.randint(10, (100, ))),
            ('user', 'buy', 'game'):
            (torch.randint(10, (100, )), torch.randint(10, (100, ))),
            ('user', 'play', 'game'): (torch.randint(10, (100, )),
                                       torch.randint(10, (100, ))),
            ('game', 'belong', 'category'): (torch.randint(10, (100, )),
                                             torch.randint(10, (100, ))),
        }), {
            'user': 10,
            'category': 10,
            'game': 10
        })
    model = GCN([16, 12, 8, 6], graph.canonical_etypes)
    print(model)

    # Test full graph
    input_embeddings1 = {
        node_name: torch.rand(10, 16)
        for node_name in ['user', 'category', 'game']
    }
    input_embeddings2 = {
        etype:
        {node_name: torch.rand(10, 16)
         for node_name in [etype[0], etype[2]]}
        for etype in graph.canonical_etypes if not etype[1].endswith('-by')
    }
    for x in [input_embeddings1, input_embeddings2]:
        outputs = model([graph] * 3, x)
        for k, v in outputs.items():
            for k2, v2 in v.items():
                print(k, k2, v2.shape)

    # Test mini-batch
    eid_dict = {
        etype: graph.edges('eid', etype=etype)
        for etype in graph.canonical_etypes
    }
    dataloader = dgl.dataloading.EdgeDataLoader(
        graph,
        eid_dict,
        dgl.dataloading.MultiLayerFullNeighborSampler(3),
        batch_size=4,
        shuffle=True,
    )
    for input_nodes, _, blocks in dataloader:
        input_embeddings1 = {
            k: torch.rand(v.size(0), 16)
            for k, v in input_nodes.items()
        }
        input_embeddings2 = {
            etype: {
                k: torch.rand(v.size(0), 16)
                for k, v in input_nodes.items() if k in [etype[0], etype[2]]
            }
            for etype in graph.canonical_etypes if not etype[1].endswith('-by')
        }

        for x in [input_embeddings1, input_embeddings2]:
            outputs = model(blocks, x)
            for k, v in outputs.items():
                for k2, v2 in v.items():
                    print(k, k2, v2.shape)

        break
