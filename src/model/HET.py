import torch
import torch.nn as nn
from model.general.additive_attention import AdditiveAttention
import dgl
from model.GCN import GCN
from model.GAT import GAT
from model.NGCF import NGCF

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HeterogeneousNetwork(torch.nn.Module):
    def __init__(self, args, graph):
        super(HeterogeneousNetwork, self).__init__()
        self.args = args
        self.graph = graph
        self.embedding = nn.ModuleDict({
            node_name: nn.Embedding(graph.num_nodes(node_name),
                                    args.node_embedding_dim)
            for node_name in graph.ntypes
        })
        if 'GCN' in args.model_name:
            self.aggregator = GCN(
                args.node_embedding_dim,
                args.node_embedding_dim,
                args.node_embedding_dim,
            )
        elif 'GAT' in args.model_name:
            self.aggregator = GAT(
                args.node_embedding_dim,
                args.node_embedding_dim // args.num_attention_heads,
                args.node_embedding_dim, args.num_attention_heads)
        elif 'NGCF' in args.model_name:
            self.aggregator = NGCF()
        else:
            raise ValueError('Unknown aggregator')

        if 'HET' in args.model_name:
            self.mask = {
                node_name: torch.tensor([
                    node_name
                    in [canonical_edge_type[0], canonical_edge_type[2]]
                    for canonical_edge_type in graph.canonical_etypes
                ])
                for node_name in graph.ntypes
            }
            self.additive_attention = AdditiveAttention(
                args.attention_query_vector_dim, args.node_embedding_dim)

    def forward(self):
        computed = {}
        for canonical_edge_type in self.graph.canonical_etypes:
            subgraph = dgl.edge_type_subgraph(self.graph,
                                              [canonical_edge_type])
            ntypes = subgraph.ntypes
            if len(ntypes) == 1:
                # src == dest
                subgraph = dgl.to_bidirected(subgraph.cpu()).to(device)
                subgraph = dgl.add_self_loop(subgraph)
                embeddings = self.aggregator(subgraph,
                                             self.embedding[ntypes[0]].weight)
                computed[(ntypes[0], canonical_edge_type)] = embeddings
            else:
                # src != dest
                subgraph = dgl.to_bidirected(
                    dgl.to_homogeneous(subgraph).cpu()).to(device)
                subgraph = dgl.add_self_loop(subgraph)
                embeddings = self.aggregator(
                    subgraph,
                    torch.cat((
                        self.embedding[ntypes[0]].weight,
                        self.embedding[ntypes[1]].weight,
                    ),
                              dim=0))
                computed[(ntypes[0], canonical_edge_type
                          )] = embeddings[:self.graph.num_nodes(ntypes[0])]
                computed[(ntypes[1], canonical_edge_type
                          )] = embeddings[self.graph.num_nodes(ntypes[0]):]

        # Don't need to aggregated multiple embedding for a node
        # if only single type of edge exists
        if 'HET' not in self.args.model_name:
            return {
                node_name:
                computed[(node_name, self.graph.canonical_etypes[0])]
                for node_name in self.graph.ntypes
            }

        # Else aggregated them
        vectors = {
            node_name: {
                canonical_edge_type:
                computed[(node_name, canonical_edge_type)] if
                (node_name, canonical_edge_type) in computed else torch.zeros(
                    self.graph.num_nodes(node_name),
                    self.args.node_embedding_dim).to(device)
                for canonical_edge_type in self.graph.canonical_etypes
            }
            for node_name in self.graph.ntypes
        }

        aggregated_vectors = {
            node_name: self.additive_attention(
                torch.stack(list(vectors[node_name].values()), dim=1),
                self.mask[node_name].expand(self.graph.num_nodes(node_name),
                                            -1))
            for node_name in self.graph.ntypes
        }

        return aggregated_vectors


if __name__ == '__main__':
    graph = dgl.heterograph({
        ('user', 'follows', 'user'):
        (torch.randint(10, (20, )), torch.randint(10, (20, ))),
        ('user', 'follows', 'topic'):
        (torch.randint(10, (20, )), torch.randint(10, (20, ))),
        ('user', 'plays', 'game'): (torch.randint(10, (20, )),
                                    torch.randint(10, (20, )))
    })
    graph = dgl.to_simple(graph)
    graph = graph.to(device)
    from parameters import parse_args
    args = parse_args()
    for x in ['HET-GCN', 'HET-GAT']:
        args.model_name = x
        model = HeterogeneousNetwork(args, graph).to(device)
        node_embeddings = model()
        print(node_embeddings)
