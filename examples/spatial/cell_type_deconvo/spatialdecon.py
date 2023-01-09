import argparse
from pprint import pprint

import numpy as np
import torch
from anndata import AnnData

from dance.data import Data
from dance.datasets.spatial import CellTypeDeconvoDatasetLite
from dance.modules.spatial.cell_type_deconvo.spatialdecon import SpatialDecon

# TODO: make this a property of the dataset class?
DATASETS = ["CARD_synthetic", "GSE174746", "SPOTLight_synthetic"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="CARD_synthetic", choices=DATASETS, help="Name of the dataset.")
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--max_iter", type=int, default=10000, help="Maximum optimization iteration.")
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
adata = AnnData(X=count_matrix, obsm={"spatial": spatial, "cell_type_portion": cell_type_portion}, dtype=np.float32)

data = Data(adata)
data.set_config(label_channel="cell_type_portion")
x, y = data.get_data(return_type="numpy")

# Initialize and train model
spaDecon = SpatialDecon(ref_count, ref_annot, ct_varname="cellType", ct_select=cell_type_portion.columns.tolist(),
                        bias=args.bias, device=device)
pred = spaDecon.fit_and_predict(x, lr=args.lr, max_iter=args.max_iter, print_period=100)

# Compute score
mse = spaDecon.score(pred, y)
print(f"mse = {mse:7.4f}")
"""To reproduce SpatialDecon benchmarks, please refer to command lines belows:

CARD synthetic
$ python spatialdecon.py --dataset CARD_synthetic --lr .01 --max_iter 2250 --bias 1

GSE174746
$ python spatialdecon.py --dataset GSE174746 --lr .0001 --max_iter 20000 --bias 1

SPOTLight synthetic
$ python spatialdecon.py --dataset SPOTLight_synthetic --lr .01 --max_iter 500 --bias 1

"""
