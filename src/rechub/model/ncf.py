import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NCF(nn.Module):
    def __init__(self, args, graph, first_num, second_num):
        super().__init__()
        self.args = args
        # Load graph for negative sampling
        self.graph = graph
        self.primary_etypes = [
            x for x in graph.canonical_etypes if not x[1].endswith('-by')
        ]
        assert args.non_graph_embedding_dim % 4 == 0
        # TODO put the following dims into parameters
        self.first_embedding = nn.ModuleDict({
            'GMF':
            nn.Embedding(first_num, args.non_graph_embedding_dim // 4),
            'MLP':
            nn.Embedding(first_num, args.non_graph_embedding_dim)
        })
        self.second_embedding = nn.ModuleDict({
            'GMF':
            nn.Embedding(second_num, args.non_graph_embedding_dim // 4),
            'MLP':
            nn.Embedding(second_num, args.non_graph_embedding_dim),
        })
        self.MLP = nn.Sequential(
            nn.Linear(2 * args.non_graph_embedding_dim,
                      args.non_graph_embedding_dim),
            nn.ReLU(),
            nn.Linear(args.non_graph_embedding_dim,
                      args.non_graph_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(args.non_graph_embedding_dim // 2,
                      args.non_graph_embedding_dim // 4),
        )
        self.final_linear = nn.Linear(args.non_graph_embedding_dim // 2,
                                      1,
                                      bias=False)

    def forward(self, first, second, task_name=None, provided_embeddings=None):
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
        first_index = first['index']
        second_index = second['index']

        # batch_size, non_graph_embedding_dim // 4
        GMF_vector = torch.mul(
            self.first_embedding['GMF'](first_index),
            self.second_embedding['GMF'](second_index),
        )
        # batch_size, non_graph_embedding_dim // 4
        MLP_vector = self.MLP(
            torch.cat((
                self.first_embedding['MLP'](first_index),
                self.second_embedding['MLP'](second_index),
            ),
                      dim=-1))
        # batch_size
        score = self.final_linear(torch.cat((GMF_vector, MLP_vector),
                                            dim=-1)).squeeze(dim=1)
        return score
