import numpy as np
import pandas as pd
import random
import time
import dgl
from model.HET import HeterogeneousNetwork
from model.NCF import NCF
from sklearn.metrics import roc_auc_score, ndcg_score
import torch
from parameters import parse_args
import os
import logging
import coloredlogs
from tqdm import tqdm
import hashlib
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
            y_pred = model(first, second)
            y_pred = y_pred.cpu().numpy()
            y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds, axis=0)

        if task['type'] == 'link-prediction(recommendation)':
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
        elif task['type'] == 'edge-attribute-regression':
            raise NotImplementedError
        else:
            raise NotImplementedError

    overall = np.average([
        np.average(list(metrics[task['name']].values())) * task['weight']
        for task in tasks
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

        graph_data = {}
        for edge in metadata['graph']['edge']:
            df = pd.read_table(
                f"./data/{args.dataset}/train/{edge['filename']}")
            graph_data[tuple(edge['scheme'])] = (torch.tensor(
                df.iloc[:, 0].values), torch.tensor(df.iloc[:, 1].values))

        graph = dgl.heterograph(graph_data, num_nodes_dict)
        for edge in metadata['graph']['edge']:
            if edge['weighted']:
                raise NotImplementedError

        graph = graph.to(device)
        model = HeterogeneousNetwork(args, graph)
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
    log_dir = f'./log/{args.model_name}-{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir,
        f"{str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')}.txt"
    )
    file_hander = logging.FileHandler(log_file_path)
    logger.addHandler(file_hander)
    coloredlogs.install(level='DEBUG',
                        logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')
    return logger


def get_train_df(task, epoch, logger):
    '''
    # TODO: what if names of the two columns are the same
    Get training dataframe with randomly negative sampling and simple cache mechanism
    '''
    if task['type'] == 'link-prediction(recommendation)':
        # Get cache filename for this epoch of training
        if args.sample_cache:
            cahe_sensitive_keys = [
                x for x in list(args.__dict__.keys()) if any([
                    y in x for y in
                    ['positive', 'negative', 'sample', 'sampling', 'cache']
                ])
            ]
            cache_sensitive_args = {
                key: args.__dict__[key]
                for key in cahe_sensitive_keys
            }
            if epoch == 1:
                logger.info(f'Cache sensitive args {cache_sensitive_args}')
            sample_cache_dir = f"./data/{args.dataset}/train/sample/{hashlib.md5((str(task)+str(cache_sensitive_args)).encode('utf-8')).hexdigest()}"
            os.makedirs(sample_cache_dir, exist_ok=True)
            cache_file_path = os.path.join(
                sample_cache_dir,
                f"{epoch % args.num_sample_cache}-{task['filename']}")

        # If cache enabled and file exists, return directly
        if args.sample_cache and os.path.isfile(cache_file_path):
            df = pd.read_table(cache_file_path)
            logger.info(f'Read cache file {cache_file_path}')
            return df

        # Else, generate it
        df_positive = pd.read_table(
            f"./data/{args.dataset}/train/{task['filename']}")
        columns = df_positive.columns.tolist()
        assert len(columns) == 2 and 'value' not in columns
        if args.strict_negative:
            positive_map = df_positive.groupby(
                columns[0]).agg(list).to_dict()[columns[1]]
        if args.positive_sampling:
            df_positive = df_positive.sample(frac=1)
            df_positive_first_based = df_positive.drop_duplicates(columns[0])
            df_positive_second_based = df_positive.drop_duplicates(columns[1])
            df_positive = pd.concat(
                [df_positive_first_based,
                 df_positive_second_based]).drop_duplicates()

        df_positive['value'] = 1

        df_negative = pd.DataFrame()
        df_negative[columns[0]] = df_positive[columns[0]]

        candidate_length = len(
            pd.read_table(f"./data/{args.dataset}/train/{columns[1]}.tsv"))

        def negative_sampling(row):
            if args.strict_negative:
                candidates = set(range(candidate_length)) - set(
                    positive_map[row[columns[0]]])
            else:
                candidates = range(candidate_length)
            new_row = [
                row[columns[0]],
                random.sample(candidates, args.negative_sampling_ratio)
            ]
            return pd.Series(new_row, index=columns)

        tqdm.pandas(desc=f"Negative sampling for task {task['name']}")
        df_negative = df_negative.progress_apply(negative_sampling, axis=1)
        df_negative = df_negative.explode(columns[1])
        df_negative[columns[1]] = df_negative[columns[1]].astype(int)
        df_negative['value'] = 0
        df = pd.concat([df_positive, df_negative])

        if args.sample_cache:
            df.to_csv(cache_file_path, sep='\t', index=False)
            logger.info(f'Write cache file {cache_file_path}')

        return df

    raise NotImplementedError


def is_graph_model():
    return any([x in args.model_name for x in ['GCN', 'GAT', 'NGCF']])
