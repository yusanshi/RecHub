import argparse
from distutils.util import strtobool


def str2bool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--non_graph_embedding_dim', type=int, default=200)
    parser.add_argument('--graph_embedding_dims',
                        type=int,
                        nargs='+',
                        default=[200, 128, 96, 64])
    parser.add_argument('--single_attribute_dim', type=int, default=40)
    parser.add_argument('--attention_query_vector_dim', type=int, default=200)
    parser.add_argument('--num_epochs_validate', type=int, default=3)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--num_attention_heads', type=int, default=10)
    parser.add_argument('--save_checkpoint', type=str2bool, default=True)

    # sampling in training
    parser.add_argument('--strict_negative', type=str2bool, default=True)
    parser.add_argument('--negative_sampling_ratio', type=int, default=4)
    parser.add_argument('--positive_sampling',
                        type=str2bool,
                        default=True,
                        help='whether to sample from multiple')
    parser.add_argument('--sample_cache',
                        type=str2bool,
                        default=True,
                        help='whether to cache training samples')
    parser.add_argument('--num_sample_cache', type=int, default=100)

    parser.add_argument(
        '--model_name',
        type=str,
        default='GCN',
        choices=[
            # Non-graph
            'NCF',

            # Graph with single type of edge (we think it as homogeneous graph)
            'GCN',
            'GAT',
            'NGCF',

            # Graph with multiple types of edge (we think it as heterogeneous graph)
            'HET-GCN',
            'HET-GAT',
            'HET-NGCF',

            # To be categorized
            'DeepFM',
            'DSSM',
            'LightGCN',
            'DiffNet',
            'DiffNet++',
            'DANSER',
            'GraphRec'
        ])
    parser.add_argument('--dataset', type=str, default='jd', choices=[
        'jd',
    ])

    args = parser.parse_args()
    return args

    # TODO: graph model aggregators: parameters for layers numbers
