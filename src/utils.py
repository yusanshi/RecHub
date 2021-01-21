import numpy as np
import pandas as pd
import time
import dgl
from model.HET import HeterogeneousNetwork
from model.NCF import NCF
from sklearn.metrics import roc_auc_score, ndcg_score
import torch
import torch.nn as nn
from parameters import parse_args
import os
import logging
import coloredlogs
import math
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()


def recall(y_true, y_score, k):
    order = np.argsort(y_score)[::-1][:k]
    return np.sum(np.take(y_true, order)) / np.sum(y_true)


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


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
    for task in tasks:
        df = pd.read_table(f"./data/{args.dataset}/{mode}/{task['filename']}")
        columns = df.columns.tolist()
        df = df.sort_values(columns[0])
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
            y_pred = model(first, second, task['name'],
                           provided_embeddings if is_graph_model() else None)
            y_pred = y_pred.cpu().numpy()
            y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds, axis=0)

        if task['type'] == 'top-k-recommendation':
            second_lengths_for_single_first = df.groupby(
                columns[0]).size().values
            assert len(
                set(second_lengths_for_single_first)
            ) == 1, f'The number of {columns[1]}s for different {columns[0]}s should be equal'
            y_trues = y_trues.reshape(-1, second_lengths_for_single_first[0])
            y_preds = y_preds.reshape(-1, second_lengths_for_single_first[0])

            # TODO AUC, recall: batch version
            metrics[task['name']] = {
                'AUC':
                np.average(
                    [roc_auc_score(x, y) for x, y in zip(y_trues, y_preds)]),
                'NDCG@5':
                ndcg_score(y_trues, y_preds, k=5, ignore_ties=True),
                'NDCG@10':
                ndcg_score(y_trues, y_preds, k=10, ignore_ties=True),
                'recall@5':
                np.average(
                    [recall(x, y, k=5) for x, y in zip(y_trues, y_preds)]),
                'recall@10':
                np.average(
                    [recall(x, y, k=10) for x, y in zip(y_trues, y_preds)])
            }
        elif task['type'] == 'interaction-attribute-regression':
            raise NotImplementedError
        else:
            raise NotImplementedError

    overall = np.average([
        np.average(list(metrics[task['name']].values())) *
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

    if is_graph_model():
        if 'HET' in args.model_name:
            assert len(
                metadata['graph']['edge']
            ) > 1, 'You have chosen a model for heterogeneous graph, but only single type of edge exists'
        else:
            assert len(
                metadata['graph']['edge']
            ) == 1, 'You have chosen a model for homogeneous graph, but more than one types of edge exists'

        if args.model_name == 'HET-GraphRec':
            assert len(metadata['graph']['edge']) == 2
            node_names = [
                metadata['graph']['edge'][0]['scheme'][0],
                metadata['graph']['edge'][0]['scheme'][2],
                metadata['graph']['edge'][1]['scheme'][0],
                metadata['graph']['edge'][1]['scheme'][2],
            ]
            assert node_names.count('user') == 3 and node_names.count(
                'item') == 3

        graph_data = {}
        if len(
            [edge['scheme'][1] for edge in metadata['graph']['edge']]) != len(
                set([edge['scheme'][1]
                     for edge in metadata['graph']['edge']])):
            raise NotImplementedError

        for edge in metadata['graph']['edge']:
            if edge['scheme'][0] == edge['scheme'][2]:
                raise NotImplementedError

            df = pd.read_table(
                f"./data/{args.dataset}/train/{edge['filename']}")
            graph_data[edge['scheme']] = (torch.tensor(df.iloc[:, 0].values),
                                          torch.tensor(df.iloc[:, 1].values))

        graph = dgl.heterograph(add_reverse(graph_data), num_nodes_dict)
        for edge in metadata['graph']['edge']:
            if edge['weighted']:
                raise NotImplementedError

        model = HeterogeneousNetwork(args, graph, metadata['task'])
        return model

    if args.model_name == 'NCF':
        assert 'user' in num_nodes_dict and 'item' in num_nodes_dict
        if len(num_nodes_dict) > 2:
            logger.warning('Nodes except user and item are ignored')
        model = NCF(args, num_nodes_dict['user'], num_nodes_dict['item'])
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
    if args.model_name in ['NCF']:
        return False
    if args.model_name in [
            'GCN', 'GAT', 'NGCF', 'HET-GCN', 'HET-GAT', 'HET-NGCF',
            'HET-GraphRec'
    ]:
        return True

    raise NotImplementedError


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        raise NotImplementedError

    def forward(self, input, target):
        pass


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
