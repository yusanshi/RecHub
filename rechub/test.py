import torch
import json

from .utils import evaluate, latest_checkpoint, create_model, create_logger, process_metadata, dict2table
from .parameters import parse_args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()


def test():
    with open(args.metadata_path) as f:
        metadata = json.load(f)
        metadata = process_metadata(metadata)
        logger.info(metadata)

    model = create_model(metadata, logger).to(device)
    checkpoint_path = latest_checkpoint(
        f'./checkpoint/{args.model_name}-{args.dataset}')
    if checkpoint_path is None:
        logger.error('No checkpoint file found!')
        exit()
    logger.info(f'Load saved parameters in {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    metrics, _ = evaluate(model, metadata['task'], 'test')
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Testing model {args.model_name} with dataset {args.dataset}')
    test()
