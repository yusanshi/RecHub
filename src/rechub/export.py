import torch
import json

from .utils import latest_checkpoint, create_model, create_logger, process_metadata, is_graph_model
from .parameters import parse_args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()


@torch.no_grad()
def export():
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
    if is_graph_model():
        input_nodes = {
            node_name: model.graph.nodes(ntype=node_name).to(device)
            for node_name in model.graph.ntypes
        }
        embeddings = model.aggregate_embeddings(
            input_nodes,
            [model.graph.to(device)] * (len(args.graph_embedding_dims) - 1))
        import ipdb
        ipdb.set_trace()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(
        f'Exporting embeddings for model {args.model_name} with dataset {args.dataset}'
    )
    export()
