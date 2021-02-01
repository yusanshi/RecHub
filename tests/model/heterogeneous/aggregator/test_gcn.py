import dgl
import torch

from rechub.model.heterogeneous.aggregator.gcn import GCN
from rechub.utils import add_reverse

graph = dgl.heterograph(
    add_reverse({
        ('user', 'follow', 'category'):
        (torch.randint(10, (100, )), torch.randint(10, (100, ))),
        ('user', 'buy', 'game'):
        (torch.randint(10, (100, )), torch.randint(10, (100, ))),
        ('user', 'play', 'game'): (torch.randint(10, (100, )),
                                   torch.randint(10, (100, ))),
        ('game', 'belong', 'category'): (torch.randint(10, (100, )),
                                         torch.randint(10, (100, ))),
    }), {
        'user': 10,
        'category': 10,
        'game': 10
    })
model = GCN([16, 12, 8, 6], graph.canonical_etypes)
print(model)

# Test full graph
input_embeddings1 = {
    node_name: torch.rand(10, 16)
    for node_name in ['user', 'category', 'game']
}
input_embeddings2 = {
    etype:
    {node_name: torch.rand(10, 16)
     for node_name in [etype[0], etype[2]]}
    for etype in graph.canonical_etypes if not etype[1].endswith('-by')
}
for x in [input_embeddings1, input_embeddings2]:
    outputs = model([graph] * 3, x)
    for k, v in outputs.items():
        for k2, v2 in v.items():
            print(k, k2, v2.shape)

# Test mini-batch
eid_dict = {
    etype: graph.edges('eid', etype=etype)
    for etype in graph.canonical_etypes
}
dataloader = dgl.dataloading.EdgeDataLoader(
    graph,
    eid_dict,
    dgl.dataloading.MultiLayerFullNeighborSampler(3),
    batch_size=4,
    shuffle=True,
)
for input_nodes, _, blocks in dataloader:
    input_embeddings1 = {
        k: torch.rand(v.size(0), 16)
        for k, v in input_nodes.items()
    }
    input_embeddings2 = {
        etype: {
            k: torch.rand(v.size(0), 16)
            for k, v in input_nodes.items() if k in [etype[0], etype[2]]
        }
        for etype in graph.canonical_etypes if not etype[1].endswith('-by')
    }

    for x in [input_embeddings1, input_embeddings2]:
        outputs = model(blocks, x)
        for k, v in outputs.items():
            for k2, v2 in v.items():
                print(k, k2, v2.shape)

    break
