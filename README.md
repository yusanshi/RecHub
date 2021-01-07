# RecHub

Implementations of some methods in recommendation.

## Models

| Model    | Full name | Type                | Paper |
| -------- | --------- | ------------------- | ----- |
| NCF      |           | Non-graph           |       |
| GCN      |           | Homogeneous graph   |       |
| GAT      |           | Homogeneous graph   |       |
| NGCF     |           | Homogeneous graph   |       |
| HET-GCN  | /         | Heterogeneous graph | /     |
| HET-GAT  | /         | Heterogeneous graph | /     |
| HET-NGCF | /         | Heterogeneous graph | /     |
|          |           |                     |       |
|          |           |                     |       |

> Note: we define the heterogeneous graph as a graph \*with different types of **edges\*** instead of a graph \*with different types of **edges or nodes\***. Thus, for a common user-item bipartite graph, although more than one types of node exist, we still think it as a homogeneous graph.

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

Basic setup.

```bash
git clone https://github.com/yusanshi/RecHub.git && cd RecHub
pip install -r requirements.txt
pip install [DGL] # [DGL] = dgl, dgl-cu92, dgl-cu101, dgl-cu102...
python -m dgl.backend.set_default_backend pytorch
```

Download the datasets.

```bash
# TODO
wget
unzip
```

Run.

```bash
# Train and save checkpoints into f'./checkpoint/{model_name}-{dataset}' directory
python src/train.py

# Load latest checkpoint and evaluate on the test set
python src/test.py
```

You can visualize the metrics with TensorBoard.

```bash
tensorboard --logdir runs
```

> Tip: by adding `REMARK` environment variable, you can make the runs name in TensorBoard more meaningful. For example, `REMARK=lr-0.001_attention-head-10 python src/train.py`.

## Development

### Use your own dataset

Follow the steps to use a custom dataset.

1. Create the dataset directory.

   Assume the name for your dataset is `foobar`. You should put the data files into `./data/foobar`. Below is a simplest example with only one type of edges for a common recommendation task:

   ```
   data
   └── foobar
       ├── test
       │   └── user-item-interact.tsv
       ├── train
       │   ├── item.tsv
       │   ├── user-item-interact.tsv
       │   └── user.tsv
       └── val
           └── user-item-interact.tsv
   ```

   `train/user-item-interact.tsv`:

   ```
   user	item
   0	0
   0	2
   0	3
   0	8
   0	12
   ......
   ```

   `val/user-item-interact.tsv` and `test/user-item-interact.tsv`:

   ```
   user	item	value
   0	45	1
   0	92	0
   0	974	0
   0	820	0
   0	530	0
   0	411	0
   0	719	0
   1	562	1
   1	941	0
   1	874	0
   1	15	0
   1	865	0
   1	145	0
   1	87	0
   ......
   ```

   `train/item.tsv`:

   ```
   item
   0
   1
   2
   3
   4
   5
   6
   ......
   ```

   `train/user.tsv`:

   ```
   user
   0
   1
   2
   3
   4
   5
   6
   ...
   ```

   Note:

   - we use TAB (`\t`) as delimiter in all files.
   - In `val/user-item-interact.tsv` and `test/user-item-interact.tsv`, the number of items for different users should be equal (in the above example, the number of items for a user is 8, with 1 positive items and 7 negative items).
   - Since we don't have any attributes for users and items, the `user.tsv` and `user.tsv` files only have one single column (the index of users and items).

2. Create the metadata file.

   Put the file named `foobar.json` into `./metadata` with the following content:

   ```json
   {
     "graph": {
       "node": [
         {
           "name": "item",
           "attribute": []
         },
         {
           "name": "user",
           "attribute": []
         }
       ],
       "edge": [
         {
           "scheme": ["user", "interact", "item"],
           "filename": "user-item-interact.tsv",
           "weighted": false
         }
       ]
     },
     "task": [
       {
         "name": "user-item-recommendation",
         "filename": "user-item-interact.tsv",
         "type": "link-prediction",
         "weight": 1
       }
     ]
   }
   ```

### Test a model

Test code for a model is in its `if __name__ == '__main__':` block. However, directly running the file will suffer from `ModuleNotFoundError`. You can solve this by adding `src` directory to the environment variable `PYTHONPATH`. Take NCF as an example:

```bash
PYTHONPATH=./src:$PYTHONPATH python src/model/NCF.py
```

### TODO

1. Support more models.
2. Wired performance at the beginning of training (e.g., GCN). Need a better weight initialization?
3. Support node attributes.
4. Support multiple tasks (e.g., `edge-attribute-regression`).
