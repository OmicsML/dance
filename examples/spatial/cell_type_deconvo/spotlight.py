import argparse
from pprint import pprint

import numpy as np
import torch
from anndata import AnnData

from dance.data import Data
from dance.datasets.spatial import CellTypeDeconvoDatasetLite
from dance.modules.spatial.cell_type_deconvo.spotlight import SPOTlight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)

# TODO: make this a property of the dataset class?
DATASETS = ["CARD_synthetic", "GSE174746", "SPOTLight_synthetic"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="CARD_synthetic", choices=DATASETS, help="Name of the dataset.")
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--rank", type=int, default=2, help="Rank of the NMF module.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--max_iter", type=int, default=4000, help="Maximum optimization iteration.")
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

ref_count, ref_annot, count_matrix, cell_type_portion, spatial = dataset.load_data()

# TODO: add ref index (or more flexible indexing option at init, e.g., as dict?) and combine with data
ref_adata = AnnData(X=ref_count, obsm={"annot": ref_annot}, dtype=np.float32)
adata = AnnData(X=count_matrix, obsm={"cell_type_portion": cell_type_portion}, dtype=np.float32)

# TODO: deprecate the need for ct_select by doing this in a preprocessing step -> convert ct into one-hot matrix
ct_select = sorted(set(ref_annot.cellType.unique().tolist()) & set(cell_type_portion.columns.tolist()))
print(f"{ct_select=}")

data = Data(adata)
data.set_config(label_channel="cell_type_portion")

# TODO: after removing ct_select, return as numpy
x, y = data.get_data(return_type="default")

model = SPOTlight(ref_count, ref_annot, "cellType", ct_select, rank=args.rank, bias=args.bias, device=device)
pred = model.fit_and_predict(x, lr=args.lr, max_iter=args.max_iter)
mse = model.score(pred, torch.FloatTensor(y[ct_select].values))

print(f"Predicted cell-type proportions of  sample 1: {pred[0].clone().cpu().numpy().round(3)}")
print(f"True cell-type proportions of  sample 1: {y.iloc[0].tolist()}")
print(f"mse = {mse:7.4f}")
"""To reproduce SpatialDecon benchmarks, please refer to command lines belows:

CARD_synthetic
$ python spotlight.py --dataset CARD_synthetic --lr .1 --max_iter 100 --rank 8 --bias 0

GSE174746
$ python spotlight.py --dataset GSE174746 --lr .1 --max_iter 15000 --rank 4 --bias 0

SPOTLight synthetic
$ python spotlight.py --dataset SPOTLight_synthetic --lr .1 --max_iter 150 --rank 10 --bias 0

"""
