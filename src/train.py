import torch
import numpy as np
import json
import os
import time
import datetime
from parameters import parse_args
from utils import EarlyStopping, evaluate, time_since, create_model, create_logger, get_train_df, is_graph_model
from torch.utils.tensorboard import SummaryWriter
import enlighten
import copy
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()


def train():
    with open(f'metadata/{args.dataset}.json') as f:
        metadata = json.load(f)
    model = create_model(metadata, logger).to(device)
    logger.info(model)

    model.eval()
    metrics, _ = evaluate(model, metadata['task'], 'val')
    model.train()
    logger.info(f'Initial metrics on validation set {metrics}')
    best_checkpoint = copy.deepcopy(model.state_dict())
    best_val_metrics = copy.deepcopy(metrics)

    criterion = torch.nn.BCEWithLogitsLoss(
    )  # TODO: different criterions based on task
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

    if not args.positive_sampling and not args.sample_cache:
        logger.warning(
            'Positive sampling is disabled, for which it will cost much time to do negative sampling. Consider enable sample cache to speedup training'
        )
    if args.positive_sampling and args.sample_cache:
        logger.warning(
            'Positive sampling and sample cache are enabled. To fully use the data, you should set `num_sample_cache` to a relatively larger value'
        )

    pbar = enlighten.get_manager().counter(total=args.num_epochs,
                                           desc='Training',
                                           unit='epochs')
    for epoch in pbar(range(1, args.num_epochs + 1)):
        loss = 0
        if is_graph_model():
            node_embeddings = model()
            for task in metadata['task']:
                if task['type'] == 'link-prediction':
                    df = get_train_df(task, epoch, logger)
                    y_pred = torch.mul(
                        node_embeddings[df.columns[0]][torch.tensor(
                            df.iloc[:, 0].values)],
                        node_embeddings[df.columns[1]][torch.tensor(
                            df.iloc[:, 1].values)],
                    ).sum(dim=-1)
                    y_true = torch.tensor(
                        df.iloc[:, 2].values).to(device).float()
                    loss += criterion(y_pred, y_true) * task['weight']
                elif task['type'] == 'edge-attribute-regression':
                    raise NotImplementedError
                else:
                    raise NotImplementedError

        elif args.model_name == 'NCF':
            assert len(metadata['task']) == 1
            task = metadata['task'][0]
            assert task['type'] == 'link-prediction'
            df = get_train_df(task, epoch, logger)
            columns = df.columns.tolist()
            assert columns == ['user', 'item', 'value']
            df = df.sort_values('user')
            train_data = np.transpose(df.values)
            train_data = torch.from_numpy(train_data).to(device)
            user_indexs, item_indexs, y_trues = train_data
            y_trues = y_trues.float()
            for i in range(math.ceil(len(df) / args.batch_size)):
                user_index = user_indexs[i * args.batch_size:(i + 1) *
                                         args.batch_size]
                item_index = item_indexs[i * args.batch_size:(i + 1) *
                                         args.batch_size]
                y_pred = model(user_index, item_index)
                y_true = y_trues[i * args.batch_size:(i + 1) * args.batch_size]
                loss += criterion(y_pred, y_true)

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

    logger.info(f'Best metrics on validation set {best_val_metrics}')
    model.load_state_dict(best_checkpoint)
    model.eval()
    metrics, _ = evaluate(model, metadata['task'], 'test')
    logger.info(f'Metrics on test set {metrics}')


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(
        f'Training model {args.model_name} with dataset {args.dataset}')
    train()
