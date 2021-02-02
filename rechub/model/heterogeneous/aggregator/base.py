import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import HeteroGraphConv
# TODO add dropout?


class HeterogeneousAggregator(nn.Module):
    def __init__(self, graph_embedding_dims, etypes):
        super().__init__()
        self.primary_etypes = [x for x in etypes if not x[1].endswith('-by')]
        assert len(graph_embedding_dims) >= 2
        self.layer_dict = nn.ModuleDict()
        for etype in self.primary_etypes:
            layers = nn.ModuleList()
            for i in range(len(graph_embedding_dims) - 2):
                layers.append(
                    HeteroGraphConv({
                        e: self.get_layer(graph_embedding_dims[i],
                                          graph_embedding_dims[i + 1], i,
                                          len(graph_embedding_dims) - 1)
                        for e in [etype[1], f'{etype[1]}-by']
                    }))
            layers.append(
                HeteroGraphConv({
                    e: self.get_layer(graph_embedding_dims[-2],
                                      graph_embedding_dims[-1],
                                      len(graph_embedding_dims) - 2,
                                      len(graph_embedding_dims) - 1)
                    for e in [etype[1], f'{etype[1]}-by']
                }))
            self.layer_dict[str(etype)] = layers
        self.reduce_dim_dict = nn.ModuleDict()
        self.node_pair2etype = {}
        for etype in self.primary_etypes:
            node_pair = tuple(sorted([etype[0], etype[2]]))
            if node_pair in self.node_pair2etype:
                self.node_pair2etype[node_pair].append(etype)
            else:
                self.node_pair2etype[node_pair] = [etype]

        for node_pair, etypes in self.node_pair2etype.items():
            if len(etypes) > 1:
                self.reduce_dim_dict[str(node_pair)] = nn.ModuleDict({
                    node_name:
                    nn.Linear(graph_embedding_dims[-1] * len(etypes),
                              graph_embedding_dims[-1])
                    for node_name in node_pair
                })

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        raise NotImplementedError

    def single_forward(self, layers, blocks, h):
        for layer, block in zip(layers, blocks):
            h = layer(block, h)
        return h

    def forward(self, blocks, input_embeddings):
        '''
        Args:
            input_embeddings: {etype: {node_name: ...}}
        Returns:
            {etype: {node_name: ...}}
        '''
        different_embeddings = True if isinstance(
            list(input_embeddings.values())[0], dict) else False
        # dgl.add_self_loop(g) # TODO
        output_embeddings = {}
        for etype in self.primary_etypes:
            if different_embeddings:
                h = input_embeddings[etype]
            else:
                h = {
                    node_name: input_embeddings[node_name]
                    for node_name in [etype[0], etype[2]]
                }
            assert len(h) == 2
            etype_layers = self.layer_dict[str(etype)]
            etype_blocks = [
                dgl.edge_type_subgraph(
                    block, [etype, (etype[2], f'{etype[1]}-by', etype[0])])
                for block in blocks
            ]
            h = self.single_forward(etype_layers, etype_blocks, h)
            assert len(h) == 2, 'Unknown error'
            output_embeddings[etype] = h

        output_embeddings = {
            node_pair: {
                node_name:
                self.reduce_dim_dict[str(node_pair)][node_name](torch.cat(
                    [output_embeddings[etype][node_name] for etype in etypes],
                    dim=-1))
                for node_name in node_pair
            } if len(etypes) > 1 else output_embeddings[etypes[0]]
            for node_pair, etypes in self.node_pair2etype.items()
        }

        return output_embeddings


# if __name__ == '__main__':
#     import torch
#     from ....utils import add_reverse
#     graph = dgl.heterograph(
#         add_reverse({
#             ('user', 'follow', 'category'):
#             (torch.randint(10, (100, )), torch.randint(10, (100, ))),
#             ('user', 'buy', 'game'):
#             (torch.randint(10, (100, )), torch.randint(10, (100, ))),
#             ('user', 'play', 'game'): (torch.randint(10, (100, )),
#                                        torch.randint(10, (100, ))),
#             ('game', 'belong', 'category'): (torch.randint(10, (100, )),
#                                              torch.randint(10, (100, ))),
#         }), {
#             'user': 10,
#             'category': 10,
#             'game': 10
#         })
#     model = GCN([16, 12, 8, 6], graph.canonical_etypes)
#     print(model)

#     # Test full graph
#     input_embeddings1 = {
#         node_name: torch.rand(10, 16)
#         for node_name in ['user', 'category', 'game']
#     }
#     input_embeddings2 = {
#         etype:
#         {node_name: torch.rand(10, 16)
#          for node_name in [etype[0], etype[2]]}
#         for etype in graph.canonical_etypes if not etype[1].endswith('-by')
#     }
#     for x in [input_embeddings1, input_embeddings2]:
#         outputs = model([graph] * 3, x)
#         for k, v in outputs.items():
#             for k2, v2 in v.items():
#                 print(k, k2, v2.shape)

#     # Test mini-batch
#     eid_dict = {
#         etype: graph.edges('eid', etype=etype)
#         for etype in graph.canonical_etypes
#     }
#     dataloader = dgl.dataloading.EdgeDataLoader(
#         graph,
#         eid_dict,
#         dgl.dataloading.MultiLayerFullNeighborSampler(3),
#         batch_size=4,
#         shuffle=True,
#     )
#     for input_nodes, _, blocks in dataloader:
#         input_embeddings1 = {
#             k: torch.rand(v.size(0), 16)
#             for k, v in input_nodes.items()
#         }
#         input_embeddings2 = {
#             etype: {
#                 k: torch.rand(v.size(0), 16)
#                 for k, v in input_nodes.items() if k in [etype[0], etype[2]]
#             }
#             for etype in graph.canonical_etypes if not etype[1].endswith('-by')
#         }

#         for x in [input_embeddings1, input_embeddings2]:
#             outputs = model(blocks, x)
#             for k, v in outputs.items():
#                 for k2, v2 in v.items():
#                     print(k, k2, v2.shape)

#         break
