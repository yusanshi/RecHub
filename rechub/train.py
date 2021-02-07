import torch
import torch.nn as nn
import dgl
import numpy as np
import pandas as pd
import json
import os
import time
import datetime
import enlighten
import copy
from torch.utils.tensorboard import SummaryWriter

from .parameters import parse_args
from .utils import EarlyStopping, evaluate, time_since, create_model, create_logger, is_graph_model, process_metadata, dict2table, deep_apply, is_single_relation_model
from .loss import BPRLoss, MarginLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()


def train():
    with open(args.metadata_path) as f:
        metadata = json.load(f)
        metadata = process_metadata(metadata)
        logger.info(metadata)

    model = create_model(metadata, logger).to(device)
    logger.info(model)

    if not args.evaluation_task_choice:
        task_to_evaluate = [x['name'] for x in metadata['task']]

    model.eval()
    metrics, _ = evaluate(
        model, [x for x in metadata['task'] if x['name'] in task_to_evaluate],
        'val')
    model.train()
    logger.info(f'Initial metrics on validation set {deep_apply(metrics)}')
    best_checkpoint_dict = {
        task['name']: copy.deepcopy(model.state_dict())
        for task in metadata['task']
    }
    best_val_metrics_dict = copy.deepcopy(metrics)

    criterions = {}

    if is_single_relation_model():
        assert len(metadata['task']) == 1

    for task in metadata['task']:
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
            if task['loss'] == 'binary-cross-entropy':
                criterions[task['name']] = nn.BCEWithLogitsLoss()
            elif task['loss'] == 'cross-entropy':
                criterions[task['name']] = nn.CrossEntropyLoss()
            elif task['loss'] == 'bpr':
                criterions[task['name']] = BPRLoss()
            elif task['loss'] == 'margin':
                criterions[task['name']] = MarginLoss()
            else:
                raise NotImplementedError

        elif task['type'] == 'interaction-attribute-regression':
            raise NotImplementedError
        else:
            raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_full = []
    early_stopping_dict = {
        task['name']: EarlyStopping(args.early_stop_patience)
        for task in metadata['task']
    }
    start_time = time.time()
    writer = SummaryWriter(
        log_dir=
        f"./runs/{args.model_name}-{args.dataset}/{str(datetime.datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')}{'-remark-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if args.save_checkpoint:
        for task in metadata['task']:
            os.makedirs(
                f"./checkpoint/{args.model_name}-{args.dataset}/{task['name']}",
                exist_ok=True)

    enlighten_manager = enlighten.get_manager()

    batch = 0
    if is_graph_model():
        etype2num_neighbors = {
            etype: np.clip(
                np.quantile(model.graph.in_degrees(etype=etype),
                            args.neighbors_sampling_quantile,
                            interpolation='nearest'),
                args.min_neighbors_sampled, args.max_neighbors_sampled)
            for etype in model.graph.canonical_etypes
        }
        logger.debug(f'Neighbors sampled {etype2num_neighbors}')

    try:
        with enlighten_manager.counter(total=args.num_epochs,
                                       desc='Training epochs',
                                       unit='epochs') as epoch_pbar:
            for epoch in epoch_pbar(range(1, args.num_epochs + 1)):
                if is_graph_model():
                    neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(
                        [etype2num_neighbors] *
                        (len(args.graph_embedding_dims) - 1))
                else:
                    neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(
                        [0])  # TODO

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

                with enlighten_manager.counter(total=len(dataloader),
                                               desc='Training batches',
                                               unit='batches',
                                               leave=False) as batch_pbar:
                    for input_nodes, positive_graph, negative_graph, blocks in batch_pbar(
                            dataloader):
                        batch += 1
                        if is_graph_model():
                            if batch == 1:
                                node_coverage = {
                                    k: len(v) / model.graph.num_nodes(k)
                                    for k, v in input_nodes.items()
                                }
                                logger.debug(
                                    f'Node coverage {deep_apply(node_coverage)}'
                                )
                            input_nodes = {
                                k: v.to(device)
                                for k, v in input_nodes.items()
                            }
                        positive_graph = positive_graph.to(device)
                        negative_graph = negative_graph.to(device)
                        if is_graph_model():
                            blocks = [block.to(device) for block in blocks]
                            output_embeddings = model.aggregate_embeddings(
                                input_nodes, blocks)
                        else:
                            output_embeddings = None
                        loss = 0
                        for task in metadata['task']:
                            positive_index = torch.stack(
                                positive_graph.edges(etype=task['scheme']))
                            negative_index = torch.stack(
                                negative_graph.edges(etype=task['scheme']))
                            index = torch.cat((positive_index, negative_index),
                                              dim=1)
                            if not is_graph_model():
                                # map indexs
                                index[0] = positive_graph.ndata[dgl.NID][
                                    task['scheme'][0]][index[0]]
                                index[1] = positive_graph.ndata[dgl.NID][
                                    task['scheme'][2]][index[1]]
                            first = {
                                'name': task['scheme'][0],
                                'index': index[0]
                            }
                            second = {
                                'name': task['scheme'][2],
                                'index': index[1]
                            }
                            y_pred = model(first, second, task['name'],
                                           output_embeddings)
                            if task['loss'] == 'binary-cross-entropy':
                                y_true = torch.cat(
                                    (torch.ones(positive_index.size(1)),
                                     torch.zeros(
                                         negative_index.size(1)))).to(device)
                                task_loss = criterions[task['name']](y_pred,
                                                                     y_true)
                            elif task['loss'] == 'cross-entropy':
                                assert torch.equal(
                                    positive_index[0].expand(
                                        args.negative_sampling_ratio,
                                        -1).transpose(0, 1),
                                    negative_index[0].view(
                                        -1, args.negative_sampling_ratio))
                                sample_length = positive_index.size(1)
                                positive_pred = y_pred[:
                                                       sample_length].unsqueeze(
                                                           dim=-1)
                                negative_pred = y_pred[sample_length:].view(
                                    sample_length,
                                    args.negative_sampling_ratio)
                                y_pred = torch.cat(
                                    (positive_pred, negative_pred), dim=1)
                                y_true = torch.zeros(sample_length).long().to(
                                    device)
                                task_loss = criterions[task['name']](y_pred,
                                                                     y_true)
                            elif task['loss'] == 'margin':
                                assert torch.equal(
                                    positive_index[0].expand(
                                        args.negative_sampling_ratio,
                                        -1).transpose(0, 1),
                                    negative_index[0].view(
                                        -1, args.negative_sampling_ratio))
                                sample_length = positive_index.size(1)
                                positive_pred = y_pred[:sample_length]
                                negative_pred = y_pred[sample_length:]
                                task_loss = criterions[task['name']](
                                    positive_pred, negative_pred)
                            else:
                                raise NotImplementedError

                            loss += task_loss * task['weight']

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

                if epoch % args.num_epochs_validate == 0:
                    model.eval()
                    metrics, overall = evaluate(model, [
                        x for x in metadata['task']
                        if x['name'] in task_to_evaluate
                    ], 'val')
                    model.train()

                    for task_name, values in metrics.items():
                        for metric, value in values.items():
                            writer.add_scalar(
                                f'Validation/{task_name}/{metric}', value,
                                epoch)
                    logger.info(
                        f"Time {time_since(start_time)}, epoch {epoch}, metrics {deep_apply(metrics)}"
                    )
                    for task_name in copy.deepcopy(task_to_evaluate):
                        early_stop, get_better = early_stopping_dict[
                            task_name](-overall[task_name])
                        if early_stop:
                            task_to_evaluate.remove(task_name)
                            logger.info(f'Task {task_name} early stopped')
                        elif get_better:
                            best_checkpoint_dict[task_name] = copy.deepcopy(
                                model.state_dict())
                            best_val_metrics_dict[task_name] = copy.deepcopy(
                                metrics[task_name])
                            if args.save_checkpoint:
                                torch.save(
                                    {'model_state_dict': model.state_dict()},
                                    f"./checkpoint/{args.model_name}-{args.dataset}/{task_name}/ckpt-{epoch}.pt"
                                )

                    if not task_to_evaluate:
                        logger.info('All tasks early stopped')
                        break

    except KeyboardInterrupt:
        logger.info('Stop in advance')

    logger.info(
        f'Best metrics on validation set\n{dict2table(best_val_metrics_dict)}')
    test_metrics_dict = {}
    for task_name, checkpoint in best_checkpoint_dict.items():
        model.load_state_dict(checkpoint)
        model.eval()
        metrics, _ = evaluate(
            model, [x for x in metadata['task'] if x['name'] == task_name],
            'test')
        test_metrics_dict[task_name] = metrics[task_name]
    logger.info(f'Metrics on test set\n{dict2table(test_metrics_dict)}')


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(
        f'Training model {args.model_name} with dataset {args.dataset}')
    train()
