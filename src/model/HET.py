import torch
import torch.nn as nn
from model.general.attention.additive import AdditiveAttention
import dgl
from model.GCN import GCN
from model.GAT import GAT
from model.NGCF import NGCF
from model.general.predictor.dnn import DNNPredictor
from model.general.predictor.dot import DotPredictor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HeterogeneousNetwork(nn.Module):
    '''
    A general module for all graph-based models.
    '''
    def __init__(self, args, graph, tasks):
        super(HeterogeneousNetwork, self).__init__()
        self.args = args
        self.graph = graph
        if args.model_name == 'GraphRec':
            pass  # TODO
        else:
            self.embedding = nn.ModuleDict({
                node_name: nn.ModuleDict({
                    str(canonical_edge_type): nn.Embedding(
                        graph.num_nodes(node_name),
                        args.graph_embedding_dims[0])
                    for canonical_edge_type in graph.canonical_etypes
                }) if args.HET_different_embeddings else nn.Embedding(
                    graph.num_nodes(node_name), args.graph_embedding_dims[0])
                for node_name in graph.ntypes
            })

        if 'GCN' in args.model_name:
            self.aggregator = GCN(args.graph_embedding_dims)
        elif 'GAT' in args.model_name:
            self.aggregator = GAT(args.graph_embedding_dims,
                                  args.num_attention_heads)
        elif 'NGCF' in args.model_name:
            self.aggregator = NGCF(args.graph_embedding_dims)
        else:
            raise NotImplementedError

        self.aggregated_embeddings = None

        if args.model_name in [
                'HET-GCN', 'HET-GAT', 'HET-NGCF', 'HET-GraphRec'
        ]:  # TODO better if contidion
            embedding_num_dict = {
                node_name: sum([
                    node_name
                    in [canonical_edge_type[0], canonical_edge_type[2]]
                    for canonical_edge_type in graph.canonical_etypes
                ])
                for node_name in graph.ntypes
            }
            final_single_embedding_dim = self.args.graph_embedding_dims[-1] * (
                self.args.num_attention_heads
                if 'GAT' in self.args.model_name else 1)

            self.predictor = nn.ModuleDict({
                task['name']: DNNPredictor(
                    args.dnn_predictor_dims
                    if args.dnn_predictor_dims[0] != 0 else [
                        final_single_embedding_dim *
                        sum(embedding_num_dict[j]
                            for j in [task['scheme'][i] for i in [0, 2]]),
                        *args.dnn_predictor_dims[1:]
                    ])
                for task in tasks
            })
        elif args.model_name in ['GCN', 'GAT', 'NGCF']:
            self.predictor = DotPredictor()
        else:
            raise NotImplementedError

    def aggregate_embeddings(self, excluded_dataframes=None):
        # TODO sample! accept some node indexs as parameters and only update related embeddings
        computed = {}
        for canonical_edge_type in self.graph.canonical_etypes:
            subgraph = dgl.edge_type_subgraph(self.graph,
                                              [canonical_edge_type])
            if excluded_dataframes is not None:
                pass  # TODO exclude positive edges from dataframes to avoid data leak
            ntypes = subgraph.ntypes
            if len(ntypes) == 1:
                raise NotImplementedError
                # src == dest
                # subgraph = dgl.to_bidirected(subgraph.cpu()).to(device)
                # embeddings = self.aggregator(subgraph,
                #                              self.embedding[ntypes[0]].weight)
                # computed[(ntypes[0], canonical_edge_type)] = embeddings
            else:
                # src != dest
                subgraph = dgl.to_bidirected(
                    dgl.to_homogeneous(subgraph).cpu()
                ).to(device)  # TODO test performance without `to_bidirected`
                embeddings = self.aggregator(
                    subgraph,
                    torch.cat(
                        ((self.embedding[ntypes[0]][str(canonical_edge_type)]
                          if self.args.HET_different_embeddings else
                          self.embedding[ntypes[0]]).weight,
                         (self.embedding[ntypes[1]][str(canonical_edge_type)]
                          if self.args.HET_different_embeddings else
                          self.embedding[ntypes[1]]).weight),
                        dim=0))
                computed[(ntypes[0], canonical_edge_type
                          )] = embeddings[:self.graph.num_nodes(ntypes[0])]
                computed[(ntypes[1], canonical_edge_type
                          )] = embeddings[self.graph.num_nodes(ntypes[0]):]

        # Else aggregated them
        self.aggregated_embeddings = {
            node_name: torch.cat([
                computed[(node_name, canonical_edge_type)]
                for canonical_edge_type in self.graph.canonical_etypes
                if (node_name, canonical_edge_type) in computed
            ],
                                 dim=-1)
            for node_name in self.graph.ntypes
        }

    def forward(self, first, second, task_name):
        '''
        Args:
            first: {
                'name': str,
                'index': (shape) batch_size
            },
            second: {
                'name': str,
                'index': (shape) batch_size
            }
        '''
        if self.aggregated_embeddings is None:
            self.aggregate_embeddings()

        if isinstance(self.predictor, nn.ModuleDict):
            predictor = self.predictor[task_name]
        else:
            predictor = self.predictor
        return predictor(
            self.aggregated_embeddings[first['name']][first['index']],
            self.aggregated_embeddings[second['name']][second['index']])


if __name__ == '__main__':
    graph = dgl.heterograph(
        {
            ('user', 'follows', 'user'):
            (torch.randint(10, (20, )), torch.randint(10, (20, ))),
            ('user', 'follows', 'topic'):
            (torch.randint(10, (20, )), torch.randint(10, (20, ))),
            ('user', 'plays', 'game'):
            (torch.randint(10, (20, )), torch.randint(10, (20, )))
        }, {
            'user': 10,
            'topic': 10,
            'game': 10
        })
    graph = dgl.to_simple(graph)
    graph = graph.to(device)
    from parameters import parse_args
    args = parse_args()
    for x in ['HET-GCN', 'HET-GAT']:
        args.model_name = x
        model = HeterogeneousNetwork(args, graph).to(device)
        first_index = torch.randint(10, (64, )).to(device)
        second_index = torch.randint(10, (64, )).to(device)
        first = {'name': 'user', 'index': first_index}
        second = {'name': 'topic', 'index': second_index}
        y_pred = model(first, second)
        print(y_pred)
