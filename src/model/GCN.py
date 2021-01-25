import torch.nn as nn
import dgl
from dgl.nn.pytorch import HeteroGraphConv, GraphConv
# TODO add dropout?


class GCN(nn.Module):
    def __init__(self, graph_embedding_dims, etypes, exchange_rate):
        super(GCN, self).__init__()
        assert len(graph_embedding_dims) >= 2
        assert all([etype[0] != etype[2]
                    for etype in etypes])  # TODO check the circle
        self.primary_etypes = [x for x in etypes if not x[1].endswith('-by')]
        etype2node = {
            etype: [etype[0], etype[2]]
            for etype in self.primary_etypes
        }
        self.node2etype = {}
        for etype, nodes in etype2node.items():
            for node in nodes:
                if node not in self.node2etype:
                    self.node2etype[node] = []
                self.node2etype[node].append(etype)
        for etypes in self.node2etype.values():
            assert len(etypes) == 2
        assert 0 <= exchange_rate <= 1
        self.exchange_rate = exchange_rate
        self.layer_dict = nn.ModuleDict()
        for etype in self.primary_etypes:
            layers = nn.ModuleList()
            for i in range(len(graph_embedding_dims) - 2):
                layers.append(
                    HeteroGraphConv({
                        e: GraphConv(graph_embedding_dims[i],
                                     graph_embedding_dims[i + 1],
                                     activation=nn.ReLU())
                        for e in [etype[1], f'{etype[1]}-by']
                    }))
            layers.append(
                HeteroGraphConv({
                    e: GraphConv(graph_embedding_dims[-2],
                                 graph_embedding_dims[-1])
                    for e in [etype[1], f'{etype[1]}-by']
                }))
            self.layer_dict[str(etype)] = layers

    def exchange(self, a, b, etypes, layer):
        p = self.exchange_rate
        etype1, etype2 = etypes
        return a * (1 - p) + b * p, a * p + b * (1 - p)

    def forward(self, blocks, input_embeddings):
        '''
        Args:
            input_embeddings: {etype: {node_name: ...}} or {node_name: ...}
        Returns:
            {etype: {node_name: ...}}
        '''
        different_embeddings = True if isinstance(
            list(input_embeddings.values())[0], dict) else False
        # dgl.add_self_loop(g) # TODO

        if different_embeddings:
            embeddings = input_embeddings
        else:
            embeddings = {
                etype: {
                    node_name: input_embeddings[node_name]
                    for node_name in [etype[0], etype[2]]
                }
                for etype in self.primary_etypes
            }

        for layer, block in enumerate(blocks):
            for etype in self.primary_etypes:
                embeddings[etype] = self.layer_dict[str(etype)][layer](
                    dgl.edge_type_subgraph(
                        block, [etype,
                                (etype[2], f'{etype[1]}-by', etype[0])]),
                    embeddings[etype],
                )
            for v in embeddings.items():
                assert len(v) == 2, 'Unknown error'
            if layer != len(blocks) - 1:
                for node, etypes in self.node2etype.items():
                    embeddings[etypes[0]][node], embeddings[
                        etypes[1]][node] = self.exchange(
                            embeddings[etypes[0]][node],
                            embeddings[etypes[1]][node], etypes, layer)

        return embeddings


if __name__ == '__main__':
    import torch
    from utils import add_reverse
    graph = dgl.heterograph(
        add_reverse({
            ('user', 'follow', 'category'):
            (torch.randint(10, (100, )), torch.randint(10, (100, ))),
            ('user', 'play', 'game'):
            (torch.randint(10, (100, )), torch.randint(10, (100, ))),
            ('game', 'belong', 'category'): (torch.randint(10, (100, )),
                                             torch.randint(10, (100, ))),
        }), {
            'user': 10,
            'category': 10,
            'game': 10
        })
    model = GCN([16, 12, 8, 6], graph.canonical_etypes, 0.5)
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
