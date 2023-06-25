import argparse
from pprint import pprint

import numpy as np
import torch

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo import DSTG
from dance.utils import set_seed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cache", action="store_true", help="Cache processed data.")
parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.AVAILABLE_DATA)
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
preprocessing_pipeline = DSTG.preprocessing_pipeline(
    n_pseudo=args.num_pseudo,
    n_top_genes=args.n_hvg,
    k_filter=args.k_filter,
    num_cc=args.num_cc,
)
dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
data = dataset.load_data(transform=preprocessing_pipeline, cache=args.cache)

(adj, x), y = data.get_data(return_type="default")
x, y = torch.FloatTensor(x), torch.FloatTensor(y.values)
adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                               torch.FloatTensor(adj.data.astype(np.int32)))
train_mask = data.get_split_mask("pseudo", return_type="torch")
inputs = (adj, x, train_mask)

# Train and evaluate model
model = DSTG(nhid=args.nhid, bias=args.bias, dropout=args.dropout, device=args.device)
pred = model.fit_predict(inputs, y, lr=args.lr, max_epochs=args.epochs, weight_decay=args.wd)
test_mask = data.get_split_mask("test", return_type="torch")
score = model.default_score_func(y[test_mask], pred[test_mask])
print(f"MSE: {score:7.4f}")
"""To reproduce DSTG benchmarks, please refer to command lines belows:

CARD synthetic $ python dstg.py --dataset CARD_synthetic --nhid 16 --lr .001 --k_filter
50

GSE174746 $ python dstg.py --dataset GSE174746 --nhid 16 --lr .0001 --k_filter 50

SPOTLight synthetic $ python dstg.py --dataset SPOTLight_synthetic --nhid 32 --lr .1
--epochs 25

"""
