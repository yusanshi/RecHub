import argparse
from distutils.util import strtobool


def str2bool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--non_graph_embedding_dim', type=int, default=200)
    parser.add_argument('--graph_embedding_dims',
                        type=int,
                        nargs='+',
                        default=[200, 128, 96, 64])
    parser.add_argument('--single_attribute_dim', type=int, default=40)
    parser.add_argument('--attention_query_vector_dim', type=int, default=200)
    parser.add_argument(
        '--dnn_predictor_dims',
        type=int,
        nargs='+',
        default=[0, 128, 1],
        help=
        'You can set first dim as 0 to make it automatically fit the input vector'
    )
    parser.add_argument('--num_epochs_validate', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)
    parser.add_argument('--different_embeddings', type=str2bool, default=True)

    # sampling in training
    parser.add_argument('--strict_negative', type=str2bool, default=True)
    parser.add_argument('--negative_sampling_ratio', type=int, default=4)
    parser.add_argument('--sample_cache',
                        type=str2bool,
                        default=True,
                        help='whether to cache training samples')
    parser.add_argument('--num_sample_cache', type=int, default=500)

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
            'HET-GraphRec',

            # To be categorized
            'DeepFM',
            'DSSM',
            'LightGCN',
            'DiffNet',
            'DiffNet++',
            'DANSER'
        ])
    parser.add_argument('--dataset',
                        type=str,
                        default='jd-small',
                        choices=[
                            'jd-small',
                            'jd-large',
                        ])
    parser.add_argument('--embedding_aggregator',
                        type=str,
                        default='concat',
                        choices=['concat', 'attn'])
    parser.add_argument('--predictor',
                        type=str,
                        default='dnn',
                        choices=['dot', 'Wdot', 'Wsdot', 'dnn'])
    parser.add_argument('--metadata_path', type=str)
    args = parser.parse_args()
    if args.metadata_path is None:
        args.metadata_path = f'metadata/{args.dataset}.json'
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
