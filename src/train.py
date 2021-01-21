import torch
import torch.nn as nn
import dgl
import numpy as np
import pandas as pd
import json
import os
import time
import datetime
from parameters import parse_args
from utils import EarlyStopping, evaluate, time_since, create_model, create_logger, is_graph_model, BPRLoss, add_scheme, dict2table, deep_apply
from torch.utils.tensorboard import SummaryWriter
import enlighten
import copy
import math
from sampler import Sampler  # TODO
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

    model.eval()
    metrics, _ = evaluate(model, metadata['task'], 'val')
    model.train()
    logger.info(f'Initial metrics on validation set {deep_apply(metrics)}')
    best_checkpoint = copy.deepcopy(model.state_dict())
    best_val_metrics = copy.deepcopy(metrics)

    # samplers = {}
    criterions = {}
    for task in metadata['task']:
        # samplers
        # samplers[task['name']] = Sampler(task, logger)

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

    # TODO
    if args.sample_cache:
        logger.warning(
            'Sample cache enabled. To fully use the data, you should set `num_sample_cache` to a relatively larger value'
        )
    enlighten_manager = enlighten.get_manager()
    epoch_pbar = enlighten_manager.counter(total=args.num_epochs,
                                           desc='Training epochs',
                                           unit='epochs')
    batch_pbar = enlighten_manager.counter(desc='Training batches',
                                           unit='batches')

    batch = 0
    try:
        for epoch in epoch_pbar(range(1, args.num_epochs + 1)):
            if is_graph_model():
                # TODO Neighbor sampling
                assert len(args.num_neighbors_sampled) == len(
                    args.graph_embedding_dims) - 1
                neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(
                    args.num_neighbors_sampled)

                # Two versions of edge_sampling, either is OK
                def edge_sampling(etype):
                    df = pd.DataFrame(
                        torch.stack(model.graph.edges(etype=etype),
                                    dim=1).numpy())
                    return df.sample(frac=1).drop_duplicates(0).index.values

                # def edge_sampling(etype):
                #     subgraph = dgl.edge_type_subgraph(model.graph, [etype])
                #     return dgl.sampling.sample_neighbors(
                #         subgraph, {
                #             etype[0]: subgraph.nodes(etype[0])
                #         },
                #         1,
                #         edge_dir='out').edata[dgl.EID]

                eid_dict = {
                    etype: edge_sampling(etype)
                    for etype in
                    model.primary_etypes  # TODO model.graph.canonical_etypes ?
                }

                # parse reverse_etypes
                etypes = copy.deepcopy(model.graph.canonical_etypes)
                reverse_etypes = {}
                for etype in model.graph.canonical_etypes:
                    if etype[1].endswith('-by'):
                        reverse_etype = (etype[2], etype[1][:-len('-by')],
                                         etype[0])
                        reverse_etypes[etype] = reverse_etype
                        reverse_etypes[reverse_etype] = etype
                        etypes.remove(etype)
                        etypes.remove(reverse_etype)
                assert len(etypes) == 0

                dataloader = dgl.dataloading.EdgeDataLoader(
                    model.graph,
                    eid_dict,
                    neighbor_sampler,
                    exclude='reverse_types',
                    reverse_etypes=reverse_etypes,
                    negative_sampler=dgl.dataloading.negative_sampler.Uniform(
                        args.negative_sampling_ratio),
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers,
                    pin_memory=True)

                batch_pbar.total = len(dataloader)
                batch_pbar.count = 0
                for input_nodes, positive_graph, negative_graph, blocks in batch_pbar(
                        dataloader):
                    batch += 1
                    if batch == 1:
                        node_coverage = {
                            k: len(v) / model.graph.num_nodes(k)
                            for k, v in input_nodes.items()
                        }
                        logger.debug(
                            f'Node coverage {deep_apply(node_coverage)}')
                    input_nodes = {
                        k: v.to(device)
                        for k, v in input_nodes.items()
                    }
                    positive_graph = positive_graph.to(device)
                    negative_graph = negative_graph.to(device)
                    blocks = [block.to(device) for block in blocks]
                    output_embeddings = model.aggregate_embeddings(
                        input_nodes, blocks)
                    loss = 0
                    for task in metadata['task']:
                        positive_index = torch.stack(
                            positive_graph.edges(etype=task['scheme']))
                        negative_index = torch.stack(
                            negative_graph.edges(etype=task['scheme']))
                        index = torch.cat((positive_index, negative_index),
                                          dim=1)
                        first = {'name': task['scheme'][0], 'index': index[0]}
                        second = {'name': task['scheme'][2], 'index': index[1]}
                        y_pred = model(first, second, task['name'],
                                       output_embeddings)
                        y_true = torch.cat(
                            (torch.ones(positive_index.size(1)),
                             torch.zeros(negative_index.size(1)))).to(device)
                        task_loss = criterions[task['name']](y_pred, y_true)
                        loss += task_loss * task['weight']['loss']

                        if len(metadata['task']) > 1:
                            writer.add_scalar(f"Train/Loss/{task['name']}",
                                              task_loss.item(), batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_full.append(loss.item())
                    writer.add_scalar('Train/Loss', loss.item(), batch)
                    if batch % args.num_batches_show_loss == 0:
                        logger.info(
                            f"Time {time_since(start_time)}, epoch {epoch}, batch {batch}, current loss {loss.item():.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                        )
            else:
                raise NotImplementedError
                # assert len(
                #     metadata['task']
                # ) == 1 and metadata['task'][0]['type'] == 'top-k-recommendation'

                # task = metadata['task'][0]
                # df = samplers[task['name']].sample(epoch)
                # columns = df.columns.tolist()
                # df = df.sort_values(columns[0])  # TODO shuffle?
                # train_data = np.transpose(df.values)
                # train_data = torch.from_numpy(train_data).to(device)
                # first_indexs, second_indexs, y_trues = train_data
                # y_trues = y_trues.float()

                # for i in range(math.ceil(len(df) / args.batch_size)):
                #     first_index = first_indexs[i * args.batch_size:(i + 1) *
                #                                args.batch_size]
                #     second_index = second_indexs[i * args.batch_size:(i + 1) *
                #                                  args.batch_size]
                #     first = {'name': columns[0], 'index': first_index}
                #     second = {'name': columns[1], 'index': second_index}
                #     y_pred = model(first, second, task['name'])
                #     y_true = y_trues[i * args.batch_size:(i + 1) * args.batch_size]
                #     loss = criterions[task['name']](y_pred, y_true)
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

            if epoch % args.num_epochs_validate == 0:
                model.eval()
                metrics, overall = evaluate(model, metadata['task'], 'val')
                model.train()
                for task_name, values in metrics.items():
                    for metric, value in values.items():
                        writer.add_scalar(f'Validation/{task_name}/{metric}',
                                          value, epoch)
                logger.info(
                    f"Time {time_since(start_time)}, epoch {epoch}, metrics {deep_apply(metrics)}"
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
    except KeyboardInterrupt:
        logger.info('Stop in advance.')

    logger.info(
        f'Best metrics on validation set\n{dict2table(best_val_metrics)}')
    model.load_state_dict(best_checkpoint)
    model.eval()
    metrics, _ = evaluate(model, metadata['task'], 'test')
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(
        f'Training model {args.model_name} with dataset {args.dataset}')
    train()
