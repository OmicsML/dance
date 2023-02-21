import argparse
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from dance.data import Data
from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo import DSTG
from dance.transforms.graph import DSTGraph
from dance.transforms.preprocess import pseudo_spatial_process
from dance.utils import set_seed
from dance.utils.matrix import normalize

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

ref_adata = data.get_split_data("ref")
ref_count = ref_adata.to_df()
ref_annot = ref_adata.obs

test_adata = data.get_split_data("test")
count_matrix = test_adata.to_df()
cell_type_portion = test_adata.obsm["cell_type_portion"]

# Set adata objects for sc ref and cell mixtures
sc_adata = AnnData(ref_count, obs=ref_annot, dtype=np.float32)
mix_adata = AnnData(count_matrix, dtype=np.float32)

# pre-process: get variable genes --> normalize --> log1p --> standardize --> out
# set scRNA to false if already using pseudo spot data with real spot data
# set to true if the reference data is scRNA (to be used for generating pseudo spots)
mix_counts, mix_labels, hvgs = pseudo_spatial_process([sc_adata, mix_adata], [ref_annot, cell_type_portion], "cellType",
                                                      args.sc_ref, args.n_hvg, args.num_pseudo)

# WARNING: features appear to have negative values, normalization does not make sense, need to check more
features = np.vstack((mix_counts[0].X, mix_counts[1].X)).astype(np.float32)
normalized_features = normalize(features, axis=1, mode="normalize")
adata = AnnData(X=normalized_features, dtype=np.float32)
adata.obsm["cell_type_portion"] = pd.concat(mix_labels).astype(np.float32).set_index(adata.obs_names)

data = Data(adata, train_size=mix_counts[0].shape[0])
DSTGraph(k_filter=args.k_filter, num_cc=args.num_cc)(data)
data.set_config(feature_channel=[None, "DSTGraph"], feature_channel_type=[None, "obsp"],
                label_channel="cell_type_portion")

(x, adj), y = data.get_data(return_type="default")
x, y = torch.FloatTensor(x), torch.FloatTensor(y.values)
adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                               torch.FloatTensor(adj.data.astype(np.int32)))
train_mask = data.get_split_mask("train", return_type="torch")

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
