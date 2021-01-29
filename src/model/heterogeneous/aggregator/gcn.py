import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
from .base import HeterogeneousAggregator


class GCN(HeterogeneousAggregator):
    def __init__(self, graph_embedding_dims, etypes):
        super().__init__(graph_embedding_dims, etypes)

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        assert 0 <= current_layer <= total_layer - 1
        if current_layer < total_layer - 1:
            return GraphConv(input_dim, output_dim, activation=nn.ReLU())

        return GraphConv(input_dim, output_dim)


if __name__ == '__main__':
    import torch
    from utils import add_reverse
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
