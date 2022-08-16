import argparse
from pprint import pprint

import torch

from dance.datasets.spatial import CellTypeDeconvoDatasetLite
from dance.modules.spatial.cell_type_deconvo.dstg import DSTGLearner

# TODO: make this a property of the dataset class?
DATASETS = ["CARD_synthetic", "GSE174746", "SPOTLight_synthetic", "toy1", "toy2"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="CARD_synthetic", choices=DATASETS, help="Name of the dataset.")
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--sc_ref", type=bool, default=True, help="Reference scRNA (True) or cell-mixtures (False).")
parser.add_argument("--N_p", type=int, default=500, help="Number of pseudo mixtures to generate.")
parser.add_argument("--n_hvg", type=int, default=2000, help="Number of HVGs.")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
parser.add_argument("--K_filter", type=int, default=0, help="Graph node filter.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--nhid", type=int, default=16, help="Number of neurons in latent layer.")
parser.add_argument("--dropout", type=float, default=0., help="Dropout rate.")
parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train the model.")
parser.add_argument("--seed", type=int, default=17, help="Random seed.")
parser.add_argument("--device", default="auto", help="Computation device.")
args = parser.parse_args()
pprint(vars(args))

# Set torch variables
torch.manual_seed(args.seed)
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

# Load dataset
dataset = CellTypeDeconvoDatasetLite(data_id=args.dataset, data_dir=args.datadir)

sc_count = dataset.data["ref_sc_count"]
sc_annot = dataset.data["ref_sc_annot"]

mix_count = dataset.data["mix_count"]
true_p = dataset.data["true_p"]

ct_select = sorted(set(sc_annot.cellType.unique().tolist()) & set(true_p.columns.tolist()))
print('ct_select =', f'{ct_select}')

ct_select_ix = sc_annot[sc_annot['cellType'].isin(ct_select)].index
sc_annot = sc_annot.loc[ct_select_ix]
sc_count = sc_count.loc[ct_select_ix]

# Initialize and train model
dstg = DSTGLearner(sc_count=sc_count, sc_annot=sc_annot, scRNA=args.sc_ref, mix_count=mix_count, clust_vr="cellType",
                   n_hvg=args.n_hvg, N_p=args.N_p, k_filter=args.K_filter, nhid=args.nhid, device=device,
                   bias=args.bias, dropout=args.dropout)

# Fit model
dstg.fit(lr=args.lr, max_epochs=args.epochs, weight_decay=args.wd)

# Predict cell-type proportions and evaluate
pred = dstg.predict()

# Compute score
mse = dstg.score(pred[args.N_p:, :], torch.Tensor(true_p[ct_select].values), 'mse')
print(f"mse = {mse:7.4f}")
"""To reproduce DSTG benchmarks, please refer to command lines belows:

CARD synthetic
$ python dstg.py --dataset CARD_synthetic --nhid 16 --lr .001

GSE174746
$ python dstg.py --dataset GSE174746 --nhid 16 --lr .0001

SPOTLight synthetic
$ python dstg.py --dataset SPOTLight_synthetic --nhid 32 --lr .1 --epochs 25

"""
