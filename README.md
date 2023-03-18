# RecHub

A library for GNN-based recommendation system.

## Models

| Model        | Full name                            | Type                | Paper                              |
| ------------ | ------------------------------------ | ------------------- | ---------------------------------- |
| NCF          | Neural Collaborative Filtering       | Non-graph           | <https://arxiv.org/abs/1708.05031> |
| GCN          | Graph Convolutional Networks         | Homogeneous graph   | <https://arxiv.org/abs/1609.02907> |
| LightGCN     | Light GCN                            | Homogeneous graph   | <https://arxiv.org/abs/2002.02126> |
| GAT          | Graph Attention Networks             | Homogeneous graph   | <https://arxiv.org/abs/1710.10903> |
| NGCF         | Neural Graph Collaborative Filtering | Homogeneous graph   | <https://arxiv.org/abs/1905.08108> |
| HET-GCN      | /                                    | Heterogeneous graph | /                                  |
| HET-LightGCN | /                                    | Heterogeneous graph | /                                  |
| HET-GAT      | /                                    | Heterogeneous graph | /                                  |
| HET-NGCF     | /                                    | Heterogeneous graph | /                                  |

Note: we define the heterogeneous graph as a graph with different types of **edges** instead of a graph with different types of **edges or nodes**. Thus, for a common user-item bipartite graph, although more than one types of node exist, we still think it as a homogeneous graph.

**WIP**

- DeepFM
- DSSM
- DiffNet
- DiffNet++
- DANSER
- GraphRec

## Requirements

- Linux-based OS
- Python 3.6+

## Get started

### Install RecHub

Install from <https://pypi.org/>:

```bash
pip install rechub
```

Or install manually:

```bash
git clone https://github.com/yusanshi/RecHub.git
cd RecHub
pip install .
```

### Install DGL

Note one of the most important dependencies for RecHub, [DGL](https://www.dgl.ai/), will not be automatically installed while installing RecHub. You should manually install CPU or CUDA build of DGL.

```bash
# This is for CPU version. For CUDA version, use dgl-cu[xxx]
pip install dgl # or dgl-cu92, dgl-cu101, dgl-cu102, dgl-cu110 for CUDA 9.2, 10.1, 10.2, 11.0, respectively.
```
Check out the instructions on <https://www.dgl.ai/pages/start.html> for more details.

If there are any problems with later commands, try to install this specific version:

```bash
pip install dgl==0.5.3 # or CUDA version: dgl-cu[xxx]==0.5.3
```

### Download the dataset

Here we use the LSEC-Small dataset used in our work [LSEC-GNN](https://github.com/yusanshi/LSEC-GNN). It is a dataset featuring live stream E-commerce.

Create an empty directory as our `ROOT_DIRECTORY`. Then:

```bash
# In ROOT_DIRECTORY
mkdir data && cd data
wget https://github.com/yusanshi/LSEC-GNN/files/6520753/LSEC-Small-aa.dummy.gz \
 https://github.com/yusanshi/LSEC-GNN/files/6520754/LSEC-Small-ab.dummy.gz \
 https://github.com/yusanshi/LSEC-GNN/files/6520757/LSEC-Small-ac.dummy.gz \
 https://github.com/yusanshi/LSEC-GNN/files/6520760/LSEC-Small-ad.dummy.gz
cat LSEC-Small-* | tar -xzvf -
```

### Write the metadata file

We use a metadata file to define the nodes, edges for the graph and the tasks. For LSEC-Small dataset, create `ROOT_DIRECTORY/metadata/LSEC.json` as follows:

```json
{
  "graph": {
    "node": [
      {
        "filename": "item.tsv",
        "attribute": []
      },
      {
        "filename": "user.tsv",
        "attribute": []
      },
      {
        "filename": "streamer.tsv",
        "attribute": []
      }
    ],
    "edge": [
      {
        "filename": "user-item-buy.tsv",
        "weighted": false
      },
      {
        "filename": "user-streamer-follow.tsv",
        "weighted": false
      },
      {
        "filename": "streamer-item-sell.tsv",
        "weighted": false
      }
    ]
  },
  "task": [
    {
      "filename": "user-item-buy.tsv",
      "type": "top-k-recommendation",
      "loss": "binary-cross-entropy",
      "weight": 1
    }
  ]
}
```

### Run

```bash
# In ROOT_DIRECTORY

# Train
python -m rechub.train \
    --dataset_path ./data/LSEC-Small/ \
    --metadata_path ./metadata/LSEC.json \
    --model_name HET-GCN \
    --embedding_aggregator concat \
    --predictor mlp \
    --edge_choice 0 1 2 \
    --save_checkpoint True \
    --checkpoint_path ./checkpoint/

# Load latest checkpoint and evaluate on the test set
python -m rechub.test \
    --dataset_path ./data/LSEC-Small/ \
    --metadata_path ./metadata/LSEC.json \
    --model_name HET-GCN \
    --embedding_aggregator concat \
    --predictor mlp \
    --edge_choice 0 1 2 \
    --checkpoint_path ./checkpoint/
```

You can visualize the metrics with TensorBoard.

```bash
tensorboard --logdir runs
```

> Tip: by adding `REMARK` environment variable, you can make the runs name in TensorBoard and log file name more meaningful. For example, `REMARK=lr-0.001_attention-head-10 python -m rechub.train ...`.

## Development

### Use your own dataset

Using LSEC-Small dataset as the example, here we demonstrate the dataset format. After this section, you can convert your own dataset into this format.

The LSEC-Small dataset captures the tripartite interaction information in live stream E-commerce scenario. We have three types of nodes: `items`, `users` and `streamers`, and three types of edges: `user-item-buy`, `user-streamer-follow` and `streamer-item-sell`. The structure of the dataset is as follows:
```bash
JD-small
├── train
│   ├── user.tsv
│   ├── item.tsv
│   ├── streamer.tsv
│   ├── user-item-buy.tsv
│   ├── user-streamer-follow.tsv
│   └── streamer-item-sell.tsv
├── val
│   └── user-item-buy.tsv
└── test
    └── user-item-buy.tsv
```

In `train`, the first three files are node description files and the last three are edge description files.

In node description files are the indexs and other attributes for nodes. In LSEC-Small dataset, there are no other attributes for nodes, but only the basic index information. So the contents of `user.tsv`, `item.tsv` and `streamer.tsv` are:
```tsv
user
0
1
2
3
4
...
```
```tsv
item
0
1
2
3
4
...
```
```tsv
streamer
0
1
2
3
4
...
```

In the edge description files, each line represents an edge. Take `user-item-buy.tsv` for example, its content is:
```tsv
user    item
0    9349
0    10535
0    19326
1    555
1    2154
...
```

In `val` and `test` directory, there are edge description files for model evaluation. Different from those in `train`, they have additional column `value` indicating the existence of the edge. For example, in `val` the content of `user-item-buy.tsv` is:
```tsv
user    item    value
1    11301    1
1    13353    1
1    15315    1
1    11318    1
1    18206    1
...
```

### TODO

1. Support more models.
2. Support node attributes.
3. Support multiple tasks (e.g., `interaction-attribute-regression`).

### Tricks

- Use this to automatically select the GPU with most free memory.

  ```
  alias python='CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) python'
  ```
