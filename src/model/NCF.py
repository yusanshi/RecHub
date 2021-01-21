import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NCF(nn.Module):
    def __init__(self, args, user_num, item_num):
        super(NCF, self).__init__()
        self.args = args
        assert args.non_graph_embedding_dim % 4 == 0
        # TODO put the following dims into parameters
        self.user_embedding = nn.ModuleDict({
            'GMF':
            nn.Embedding(user_num, args.non_graph_embedding_dim // 4),
            'MLP':
            nn.Embedding(user_num, args.non_graph_embedding_dim)
        })
        self.item_embedding = nn.ModuleDict({
            'GMF':
            nn.Embedding(item_num, args.non_graph_embedding_dim // 4),
            'MLP':
            nn.Embedding(item_num, args.non_graph_embedding_dim),
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
        assert first['name'] == 'user' and second['name'] == 'item'
        user_index = first['index']
        item_index = second['index']

        # batch_size, non_graph_embedding_dim // 4
        GMF_vector = torch.mul(
            self.user_embedding['GMF'](user_index),
            self.item_embedding['GMF'](item_index),
        )
        # batch_size, non_graph_embedding_dim // 4
        MLP_vector = self.MLP(
            torch.cat((
                self.user_embedding['MLP'](user_index),
                self.item_embedding['MLP'](item_index),
            ),
                      dim=-1))
        # batch_size
        score = self.final_linear(torch.cat((GMF_vector, MLP_vector),
                                            dim=-1)).squeeze(dim=1)
        return score


if __name__ == '__main__':
    from parameters import parse_args
    args = parse_args()
    args.model_name = 'NCF'
    model = NCF(args, 100, 100).to(device)
    user_index = torch.randint(100, (64, )).to(device)
    item_index = torch.randint(100, (64, )).to(device)
    first = {'name': 'user', 'index': user_index}
    second = {'name': 'item', 'index': item_index}
    y_pred = model(first, second)
    print(y_pred)
