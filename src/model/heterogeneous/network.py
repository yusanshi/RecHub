import torch
import torch.nn as nn
from ..general.attention import AdditiveAttention
import dgl
from .aggregator import GCN, GAT, NGCF
from ..general.predictor import DNNPredictor, DotPredictor, WDotPredictor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HeterogeneousNetwork(nn.Module):
    '''
    A general module for all graph-based models.
    '''
    def __init__(self, args, graph, tasks):
        super().__init__()
        self.args = args
        self.graph = graph
        self.primary_etypes = [
            x for x in graph.canonical_etypes if not x[1].endswith('-by')
        ]
        if args.different_embeddings:
            self.embedding = nn.ModuleDict({
                str(etype): nn.ModuleDict({
                    node_name: nn.Embedding(graph.num_nodes(node_name),
                                            args.graph_embedding_dims[0])
                    for node_name in [etype[0], etype[2]]
                })
                for etype in self.primary_etypes
            })
        else:
            self.embedding = nn.ModuleDict({
                node_name: nn.Embedding(graph.num_nodes(node_name),
                                        args.graph_embedding_dims[0])
                for node_name in graph.ntypes
            })

        if 'GCN' in args.model_name:
            self.aggregator = GCN(args.graph_embedding_dims,
                                  graph.canonical_etypes)
        elif 'GAT' in args.model_name:
            self.aggregator = GAT(args.graph_embedding_dims,
                                  graph.canonical_etypes,
                                  args.num_attention_heads)
        elif 'NGCF' in args.model_name:
            self.aggregator = NGCF(args.graph_embedding_dims,
                                   graph.canonical_etypes)
        else:
            raise NotImplementedError

        final_single_embedding_dim = self.args.graph_embedding_dims[-1] * (
            self.args.num_attention_heads
            if 'GAT' in self.args.model_name else 1)

        if args.embedding_aggregator == 'concat':
            embedding_num_dict = {
                node_name: sum([
                    node_name in [etype[0], etype[2]]
                    for etype in self.primary_etypes
                ])
                for node_name in graph.ntypes
            }
            final_embedding_dim_dict = {
                task['name']: (final_single_embedding_dim *
                               embedding_num_dict[task['scheme'][0]],
                               final_single_embedding_dim *
                               embedding_num_dict[task['scheme'][2]])
                for task in tasks
            }
        elif args.embedding_aggregator == 'attn':
            final_embedding_dim_dict = {
                task['name']:
                (final_single_embedding_dim, final_single_embedding_dim)
                for task in tasks
            }
            self.additive_attention = AdditiveAttention(
                args.attention_query_vector_dim, final_single_embedding_dim)
        else:
            raise NotImplementedError

        if args.predictor == 'dot':
            self.predictor = DotPredictor()
        elif args.predictor == 'Wdot':
            # TODO what if two nodes belong to the same type
            self.predictor = nn.ModuleDict({
                task['name']:
                WDotPredictor(final_embedding_dim_dict[task['name']],
                              min(final_embedding_dim_dict[task['name']]))
                for task in tasks
            })
        elif args.predictor == 'dnn':
            self.predictor = nn.ModuleDict({
                task['name']:
                DNNPredictor(args.dnn_predictor_dims
                             if args.dnn_predictor_dims[0] != 0 else [
                                 sum(final_embedding_dim_dict[task['name']]
                                     ), *args.dnn_predictor_dims[1:]
                             ])
                for task in tasks
            })
            # import ipdb
            # ipdb.set_trace()
        else:
            raise NotImplementedError

    def aggregate_embeddings(self, input_nodes, blocks):
        if self.args.different_embeddings:
            input_embeddings = {
                etype: {
                    node_name: self.embedding[str(etype)][node_name](
                        input_nodes[node_name])
                    for node_name in [etype[0], etype[2]]
                }
                for etype in self.primary_etypes
            }
        else:
            input_embeddings = {
                node_name: self.embedding[node_name](input_nodes[node_name])
                for node_name in self.graph.ntypes
            }

        output_embeddings = self.aggregator(blocks, input_embeddings)
        # transpose the nested dict
        output_embeddings = {
            node_name: [
                output_embeddings[etype][node_name]
                for etype in output_embeddings.keys()
                if node_name in output_embeddings[etype]
            ]
            for node_name in self.graph.ntypes
        }

        if self.args.embedding_aggregator == 'concat':

            def embedding_aggregator(x):
                return torch.cat(x, dim=-1)
        elif self.args.embedding_aggregator == 'attn':

            def embedding_aggregator(x):
                return self.additive_attention(torch.stack(x, dim=1))

        output_embeddings = {
            k: embedding_aggregator(v)
            for k, v in output_embeddings.items()
        }
        return output_embeddings

    def forward(self, first, second, task_name, provided_embeddings):
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
        assert provided_embeddings is not None

        if isinstance(self.predictor, nn.ModuleDict):
            predictor = self.predictor[task_name]
        else:
            predictor = self.predictor

        return predictor(provided_embeddings[first['name']][first['index']],
                         provided_embeddings[second['name']][second['index']])


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
