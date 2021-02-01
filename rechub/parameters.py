import argparse
from distutils.util import strtobool


def str2bool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--non_graph_embedding_dim', type=int, default=200)
    parser.add_argument('--graph_embedding_dims',
                        type=int,
                        nargs='+',
                        default=[200, 128, 64])
    parser.add_argument(
        '--neighbors_sampling_quantile',
        type=float,
        default=0.9,
        help=
        'Set the number of sampled neighbors to the quantile of the numbers of neighbors'
    )
    parser.add_argument('--min_neighbors_sampled', type=int, default=4)
    parser.add_argument('--max_neighbors_sampled', type=int, default=512)
    parser.add_argument('--single_attribute_dim', type=int,
                        default=40)  # TODO: support attributes
    parser.add_argument('--attention_query_vector_dim', type=int, default=200)
    parser.add_argument(
        '--dnn_predictor_dims',
        type=int,
        nargs='+',
        default=[0, 128, 1],
        help=
        'You can set first dim as 0 to make it automatically fit the input vector'
    )
    parser.add_argument('--num_batches_show_loss', type=int, default=50)
    parser.add_argument('--num_epochs_validate', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)
    parser.add_argument('--different_embeddings', type=str2bool, default=True)
    parser.add_argument('--negative_sampling_ratio', type=int, default=4)

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
            'GraphRec',
            'DeepFM',
            'DSSM',
            'LightGCN',
            'DiffNet',
            'DiffNet++',
            'DANSER'
        ])
    parser.add_argument('--dataset', type=str, default='jd')
    parser.add_argument('--embedding_aggregator',
                        type=str,
                        default='concat',
                        choices=['concat', 'attn'])
    parser.add_argument('--predictor',
                        type=str,
                        default='dnn',
                        choices=['dot', 'Wdot', 'dnn'])
    parser.add_argument('--metadata_path', type=str)
    parser.add_argument('--node_choice',
                        type=int,
                        nargs='+',
                        default=[],
                        help='Left empty to use all in metadata file')
    parser.add_argument('--edge_choice',
                        type=int,
                        nargs='+',
                        default=[],
                        help='Left empty to use all in metadata file')
    parser.add_argument('--task_choice',
                        type=int,
                        nargs='+',
                        default=[],
                        help='Left empty to use all in metadata file')
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(
            'Warning: if you are not in testing mode, you may have got some parameters wrong input'
        )
    if args.metadata_path is None:
        args.metadata_path = f'metadata/{args.dataset}.json'
    return args
