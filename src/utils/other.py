import numpy as np
import pandas as pd
import time
import dgl
from model import *  # TODO: how to import relatively
import torch
from parameters import parse_args  # TODO: how to import relatively
import os
import logging
import coloredlogs
import math
import datetime
import copy
from .metrics import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()

# A simple cache mechanism for df reading and sorting, since it will be run for many times
_df_cache_for_validation = {}


@torch.no_grad()
def evaluate(model, tasks, mode):
    metrics = {}

    if is_graph_model():
        input_nodes = {
            node_name: model.graph.nodes(ntype=node_name).to(device)
            for node_name in model.graph.ntypes
        }
        provided_embeddings = model.aggregate_embeddings(
            input_nodes,
            [model.graph.to(device)] * (len(args.graph_embedding_dims) - 1))
    else:
        provided_embeddings = None

    for task in tasks:
        file_path = f"./data/{args.dataset}/{mode}/{task['filename']}"
        if mode == 'val' and file_path in _df_cache_for_validation:
            df = _df_cache_for_validation[file_path]
        else:
            df = pd.read_table(file_path)
            df.sort_values(df.columns[0], inplace=True)
            if mode == 'val':
                _df_cache_for_validation[file_path] = df

        columns = df.columns.tolist()
        test_data = np.transpose(df.values)
        test_data = torch.from_numpy(test_data).to(device)
        first_indexs, second_indexs, y_trues = test_data
        y_preds = []
        y_trues = y_trues.cpu().numpy()
        for i in range(math.ceil(len(df) / (8 * args.batch_size))):
            first_index = first_indexs[i * (8 * args.batch_size):(i + 1) *
                                       (8 * args.batch_size)]
            second_index = second_indexs[i * (8 * args.batch_size):(i + 1) *
                                         (8 * args.batch_size)]
            first = {'name': columns[0], 'index': first_index}
            second = {'name': columns[1], 'index': second_index}
            y_pred = model(first, second, task['name'], provided_embeddings)
            y_pred = y_pred.cpu().numpy()
            y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds, axis=0)

        if task['type'] == 'top-k-recommendation':
            single_sample_length = df.groupby(columns[0]).size().values
            assert len(
                set(single_sample_length)
            ) == 1, f'The number of {columns[1]}s for different {columns[0]}s should be equal'
            y_trues = y_trues.reshape(-1, single_sample_length[0])
            y_preds = y_preds.reshape(-1, single_sample_length[0])
            metrics[task['name']] = {
                'AUC': fast_roc_auc_score(y_trues, y_preds),
                'MRR': mrr(y_trues, y_preds),
                'NDCG@10': ndcg_score(y_trues, y_preds, k=10,
                                      ignore_ties=True),
                'NDCG@50': ndcg_score(y_trues, y_preds, k=50,
                                      ignore_ties=True),
                'Recall@10': recall(y_trues, y_preds, k=10),
                'Recall@50': recall(y_trues, y_preds, k=50),
            }
        elif task['type'] == 'interaction-attribute-regression':
            raise NotImplementedError
        else:
            raise NotImplementedError
    overall = np.mean([
        np.mean(list(metrics[task['name']].values())) *
        task['weight']['metric'] for task in tasks
    ])
    return metrics, overall


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory) if 'keep' not in x
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def create_model(metadata, logger):
    num_nodes_dict = {
        node['name']:
        len(pd.read_table(f"./data/{args.dataset}/train/{node['name']}.tsv"))
        for node in metadata['graph']['node']
    }
    for node in metadata['graph']['node']:
        if len(node['attribute']) != 0:
            logger.warning(
                f"The attributes of node {node['name']} are ignored")

    if is_single_relation_model():
        assert len(metadata['graph']['edge']) == 1
    else:
        assert len(metadata['graph']['edge']) > 1

    # TODO
    # if args.model_name == 'HET-GraphRec':
    #     assert len(metadata['graph']['edge']) == 2
    #     node_names = [
    #         metadata['graph']['edge'][0]['scheme'][0],
    #         metadata['graph']['edge'][0]['scheme'][2],
    #         metadata['graph']['edge'][1]['scheme'][0],
    #         metadata['graph']['edge'][1]['scheme'][2],
    #     ]
    #     assert node_names.count('user') == 3 and node_names.count(
    #         'item') == 3

    graph_data = {}
    if len([edge['scheme'][1] for edge in metadata['graph']['edge']]) != len(
            set([edge['scheme'][1] for edge in metadata['graph']['edge']])):
        raise NotImplementedError

    for edge in metadata['graph']['edge']:
        if edge['scheme'][0] == edge['scheme'][2]:
            raise NotImplementedError

        df = pd.read_table(f"./data/{args.dataset}/train/{edge['filename']}")
        graph_data[edge['scheme']] = (torch.as_tensor(df.iloc[:, 0].values),
                                      torch.as_tensor(df.iloc[:, 1].values))

    graph = dgl.heterograph(add_reverse(graph_data), num_nodes_dict)
    if is_graph_model():
        for edge in metadata['graph']['edge']:
            if edge['weighted']:
                raise NotImplementedError
        model = HeterogeneousNetwork(args, graph, metadata['task'])
        return model

    if args.model_name == 'NCF':
        assert 'user' in num_nodes_dict and 'item' in num_nodes_dict
        if len(num_nodes_dict) > 2:
            logger.warning('Nodes except user and item are ignored')
        model = NCF(args, graph, num_nodes_dict['user'],
                    num_nodes_dict['item'])
        return model

    raise NotImplementedError(
        f'This model {args.model_name} is under development')


def create_logger():
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG',
                        logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')
    log_dir = f'./log/{args.model_name}-{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir,
        f"{str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')}.txt"
    )
    logger.info(f'Check {log_file_path} for the log of this run')
    file_hander = logging.FileHandler(log_file_path)
    logger.addHandler(file_hander)
    return logger


def is_graph_model():
    '''
    Whether need the message passing on the graph
    '''
    if args.model_name in ['NCF']:
        return False
    if args.model_name in [
            'GCN', 'GAT', 'NGCF', 'HET-GCN', 'HET-GAT', 'HET-NGCF', 'GraphRec'
    ]:
        return True

    raise NotImplementedError


def is_single_relation_model():
    if args.model_name in ['NCF', 'GCN', 'GAT', 'NGCF']:
        return True
    if args.model_name in ['HET-GCN', 'HET-GAT', 'HET-NGCF']:
        return False

    raise NotImplementedError


def add_scheme(metadata):
    def parse_scheme_from_filename(filename):
        filename = filename.split('.')[0].split('-')
        assert len(filename) == 3
        return tuple(filename[x] for x in [0, 2, 1])

    for edge in metadata['graph']['edge']:
        edge['scheme'] = parse_scheme_from_filename(edge['filename'])
    for task in metadata['task']:
        task['scheme'] = parse_scheme_from_filename(task['filename'])

    return metadata


def add_reverse(graph_data):
    '''
    Add reverse edges for graph data before feed into `dgl.heterograph`
    '''
    for scheme in list(graph_data.keys()):
        if scheme[0] == scheme[2]:
            graph_data[scheme] = (
                torch.cat(graph_data[scheme]),
                torch.cat(graph_data[scheme][::-1]),
            )
        else:
            reversed_scheme = (scheme[2], f'{scheme[1]}-by', scheme[0])
            graph_data[reversed_scheme] = (graph_data[scheme][1],
                                           graph_data[scheme][0])
    return graph_data


def copy_arguments(f):
    def selectively_copy(x):
        if isinstance(x, list) or isinstance(x, dict):
            return copy.deepcopy(x)
        else:
            return x

    def wrapper(*args, **kwargs):
        args = tuple(selectively_copy(x) for x in args)
        kwargs = {k: selectively_copy(v) for k, v in kwargs.items()}
        return f(*args, **kwargs)

    return wrapper


@copy_arguments
def deep_apply(d, f=lambda x: f'{x:.4f}'):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = deep_apply(v, f)
        else:
            d[k] = f(v)
    return d


def dict2table(d, k_fn=str, v_fn=lambda x: f'{x:.4f}'):
    '''
    Convert a nested dict to markdown table
    '''
    def parse_header(d, depth=0):
        if isinstance(list(d.values())[0], dict):
            header = parse_header(list(d.values())[0], depth=depth + 1)
            for v in d.values():
                assert header == parse_header(v, depth=depth + 1)
            return header
        else:
            return f"| {' | '.join([''] * depth + list(map(k_fn, d.keys())))} |"

    def parse_content(d, accumulated_keys=[]):
        if isinstance(list(d.values())[0], dict):
            contents = []
            for k, v in d.items():
                contents.extend(parse_content(v, accumulated_keys + [k_fn(k)]))
            return contents
        else:
            return [
                f"| {' | '.join(accumulated_keys + list(map(v_fn, d.values())))} |"
            ]

    lines = [parse_header(d), *parse_content(d)]
    return '\n'.join(lines)
