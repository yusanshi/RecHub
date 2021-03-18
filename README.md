# RecHub

Implementations of some methods in recommendation.

## Models

| Model        | Full name | Type                | Paper |
| ------------ | --------- | ------------------- | ----- |
| NCF          |           | Non-graph           |       |
| GCN          |           | Homogeneous graph   |       |
| LightGCN     |           | Homogeneous graph   |       |
| GAT          |           | Homogeneous graph   |       |
| NGCF         |           | Homogeneous graph   |       |
| HET-GCN      | /         | Heterogeneous graph | /     |
| HET-LightGCN | /         | Heterogeneous graph | /     |
| HET-GAT      | /         | Heterogeneous graph | /     |
| HET-NGCF     | /         | Heterogeneous graph | /     |
|              |           |                     |       |
|              |           |                     |       |

> Note: we define the heterogeneous graph as a graph with different types of **edges** instead of a graph with different types of **edges or nodes**. Thus, for a common user-item bipartite graph, although more than one types of node exist, we still think it as a homogeneous graph.

**WIP**

- DeepFM
- DSSM
- LightGCN
- DiffNet
- DiffNet++
- DANSER
- GraphRec
-
-

## Requirements

- Linux-based OS
- Python 3.6+

## Get started

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

Download the datasets.

```bash
# TODO
wget
unzip
```

Run.

```bash
# Train
python -m rechub.train \
    --dataset_path ./data/JD-small/ \
    --metadata_path ./metadata/jd.json \
    --model_name HET-GCN \
    --embedding_aggregator concat \
    --predictor mlp \
    --edge_choice 0 1 2 \
    --save_checkpoint True \
    --checkpoint_path ./checkpoint/

# Load latest checkpoint and evaluate on the test set
python -m rechub.test \
    --dataset_path ./data/JD-small/ \
    --metadata_path ./metadata/jd.json \
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

TODO

### Test a model

TODO

### TODO

1. Support more models.
2. Support node attributes.
3. Support multiple tasks (e.g., `interaction-attribute-regression`).

### Tricks

- Use this to automatically select the GPU with most free memory.

  ```
  alias python='CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) python'
  ```
