import argparse
from pprint import pprint

import numpy as np
import scanpy as sc
import torch

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo import DSTG
from dance.transforms import AnnDataTransform, FilterGenesCommon, PseudoMixture, RemoveSplit, SetConfig
from dance.transforms.graph import DSTGraph
from dance.utils import set_seed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cache", action="store_true", help="Cache processed data.")
parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.DATASETS)
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--sc_ref", type=bool, default=True, help="Reference scRNA (True) or cell-mixtures (False).")
parser.add_argument("--num_pseudo", type=int, default=500, help="Number of pseudo mixtures to generate.")
parser.add_argument("--n_hvg", type=int, default=2000, help="Number of HVGs.")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
parser.add_argument("--k_filter", type=int, default=200, help="Graph node filter.")
parser.add_argument("--num_cc", type=int, default=30, help="Dimension of canonical correlation analysis.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--nhid", type=int, default=16, help="Number of neurons in latent layer.")
parser.add_argument("--dropout", type=float, default=0., help="Dropout rate.")
parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train the model.")
parser.add_argument("--seed", type=int, default=17, help="Random seed.")
parser.add_argument("--device", default="auto", help="Computation device.")
args = parser.parse_args()
set_seed(args.seed)
pprint(vars(args))

# Load dataset
dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
data = dataset.load_data()

FilterGenesCommon(split_keys=["ref", "test"], log_level="INFO")(data)
PseudoMixture(n_pseudo=args.num_pseudo, out_split_name="pseudo")(data)
RemoveSplit(split_name="ref", log_level="INFO")(data)
AnnDataTransform(sc.pp.normalize_total, target_sum=1e4)(data)
AnnDataTransform(sc.pp.log1p)(data)
AnnDataTransform(sc.pp.highly_variable_genes, flavor="seurat", n_top_genes=args.n_hvg, batch_key="batch",
                 subset=True)(data)
AnnDataTransform(sc.pp.normalize_total, target_sum=1)(data)
DSTGraph(k_filter=args.k_filter, num_cc=args.num_cc, ref_split="pseudo", inf_split="test")(data)
SetConfig({
    "feature_channel": [None, "DSTGraph"],
    "feature_channel_type": ["X", "obsp"],
    "label_channel": "cell_type_portion"
})(data)

(x, adj), y = data.get_data(return_type="default")
x, y = torch.FloatTensor(x), torch.FloatTensor(y.values)
adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                               torch.FloatTensor(adj.data.astype(np.int32)))
train_mask = data.get_split_mask("pseudo", return_type="torch")

# Train and evaluate model
model = DSTG(nhid=args.nhid, bias=args.bias, dropout=args.dropout, device=args.device)
pred = model.fit_and_predict(x, adj, y, train_mask, lr=args.lr, max_epochs=args.epochs, weight_decay=args.wd)
mse = model.score(pred[data.test_idx], y[data.test_idx], "mse")
print(f"mse = {mse:7.4f}")
"""To reproduce DSTG benchmarks, please refer to command lines belows:

CARD synthetic
$ python dstg.py --dataset CARD_synthetic --nhid 16 --lr .001 --k_filter 50

GSE174746
$ python dstg.py --dataset GSE174746 --nhid 16 --lr .0001 --k_filter 50

SPOTLight synthetic
$ python dstg.py --dataset SPOTLight_synthetic --nhid 32 --lr .1 --epochs 25

"""
