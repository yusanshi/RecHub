import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
import datetime
from parameters import parse_args
from utils import EarlyStopping, evaluate, time_since, create_model, create_logger, is_graph_model, BPRLoss, add_scheme, dict2table
from torch.utils.tensorboard import SummaryWriter
import enlighten
import copy
import math
from sampler import Sampler
import functools
import operator
from itertools import chain

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()


def train():
    with open(args.metadata_path) as f:
        metadata = json.load(f)
        metadata = add_scheme(metadata)

    assert set([node['name'] for node in metadata['graph']['node']]) == set(
        chain.from_iterable([
            [edge['scheme'][0], edge['scheme'][2]]
            for edge in metadata['graph']['edge']
        ])), 'Node type differs between node metadata and edge metadata'

    assert set([task['filename'] for task in metadata['task']]) <= set([
        edge['filename'] for edge in metadata['graph']['edge']
    ]), 'There are files in task metadata but not in graph edge metadata'

    model = create_model(metadata, logger).to(device)
    logger.info(model)

    if args.model_name in ['NCF', 'HET-GraphRec']:
        assert len(
            metadata['task']
        ) == 1 and metadata['task'][0]['type'] == 'top-k-recommendation'

    model.eval()
    metrics, _ = evaluate(model, metadata['task'], 'val')
    model.train()
    logger.info(f'Initial metrics on validation set {metrics}')
    best_checkpoint = copy.deepcopy(model.state_dict())
    best_val_metrics = copy.deepcopy(metrics)

    samplers = {}
    criterions = {}
    for task in metadata['task']:
        # samplers
        samplers[task['name']] = Sampler(task, logger)

        # criterions
        if task['type'] == 'top-k-recommendation':
            original_loss_map = {
                'NGCF': 'bpr',
                'HET-NGCF': 'bpr',
                # TODO
            }
            if args.model_name in original_loss_map and task[
                    'loss'] != original_loss_map[args.model_name]:
                logger.warning(
                    'You are using a different type of loss with the type in the paper'
                )
            if task['loss'] == 'log':
                criterions[task['name']] = nn.BCEWithLogitsLoss()
            elif task['loss'] == 'bpr':
                criterions[task['name']] = BPRLoss()
            else:
                raise NotImplementedError

        elif task['type'] == 'interaction-attribute-regression':
            raise NotImplementedError
        else:
            raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_full = []
    early_stopping = EarlyStopping(args.early_stop_patience)
    start_time = time.time()
    writer = SummaryWriter(
        log_dir=
        f"./runs/{args.model_name}-{args.dataset}/{str(datetime.datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if args.save_checkpoint:
        os.makedirs(f'./checkpoint/{args.model_name}-{args.dataset}',
                    exist_ok=True)

    if args.sample_cache:
        logger.warning(
            'Sample cache enabled. To fully use the data, you should set `num_sample_cache` to a relatively larger value'
        )

    pbar = enlighten.get_manager().counter(total=args.num_epochs,
                                           desc='Training',
                                           unit='epochs')
    for epoch in pbar(range(1, args.num_epochs + 1)):
        losses = {task['name']: 0 for task in metadata['task']}
        dataframes = {
            task['name']: samplers[task['name']].sample(epoch)
            for task in metadata['task']
        }

        if is_graph_model():
            model.aggregate_embeddings(dataframes)

        for task in metadata['task']:
            df = dataframes[task['name']]
            columns = df.columns.tolist()
            df = df.sort_values(columns[0])  # TODO shuffle?
            train_data = np.transpose(df.values)
            train_data = torch.from_numpy(train_data).to(device)
            first_indexs, second_indexs, y_trues = train_data
            y_trues = y_trues.float()

            for i in range(math.ceil(len(df) / args.batch_size)):
                first_index = first_indexs[i * args.batch_size:(i + 1) *
                                           args.batch_size]
                second_index = second_indexs[i * args.batch_size:(i + 1) *
                                             args.batch_size]
                first = {'name': columns[0], 'index': first_index}
                second = {'name': columns[1], 'index': second_index}
                y_pred = model(first, second, task['name'])
                y_true = y_trues[i * args.batch_size:(i + 1) * args.batch_size]
                losses[task['name']] += criterions[task['name']](y_pred,
                                                                 y_true)

                # TODO backword in a batch instead of an epoch

        if len(metadata['task']) > 1:
            for task in metadata['task']:
                writer.add_scalar(f"Train/Loss/{task['name']}",
                                  losses[task['name']].item(), epoch)
        loss = functools.reduce(operator.add, [
            losses[task['name']] * task['weight']['loss']
            for task in metadata['task']
        ])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_full.append(loss.item())
        writer.add_scalar('Train/Loss', loss.item(), epoch)
        logger.info(
            f"Time {time_since(start_time)}, epoch {epoch}, current loss {loss.item():.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
        )
        if epoch % args.num_epochs_validate == 0:
            model.eval()
            metrics, overall = evaluate(model, metadata['task'], 'val')
            model.train()
            for task_name, values in metrics.items():
                for metric, value in values.items():
                    writer.add_scalar(f'Validation/{task_name}/{metric}',
                                      value, epoch)
            logger.info(
                f"Time {time_since(start_time)}, epoch {epoch}, metrics {metrics}"
            )
            early_stop, get_better = early_stopping(-overall)
            if early_stop:
                logger.info('Early stop.')
                break
            elif get_better:
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_val_metrics = copy.deepcopy(metrics)
                if args.save_checkpoint:
                    torch.save({
                        'model_state_dict': model.state_dict()
                    }, f"./checkpoint/{args.model_name}-{args.dataset}/ckpt-{epoch}.pt"
                               )

    logger.info(
        f"Best metrics on validation set\n{dict2table(best_val_metrics, v_fn=lambda x: f'{x:.4f}')}"
    )
    model.load_state_dict(best_checkpoint)
    model.eval()
    metrics, _ = evaluate(model, metadata['task'], 'test')
    logger.info(
        f"Metrics on test set\n{dict2table(metrics, v_fn=lambda x: f'{x:.4f}')}"
    )


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(
        f'Training model {args.model_name} with dataset {args.dataset}')
    train()
