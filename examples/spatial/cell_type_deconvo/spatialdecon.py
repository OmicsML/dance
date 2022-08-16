import os.path as o
import sys

#use if running from dance/examples/spatial/cell_type_deconvo
root_path = o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../../.."))
sys.path.append(root_path)

import argparse
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dance.datasets.spatial import CellTypeDeconvoDatasetLite
from dance.modules.spatial.cell_type_deconvo.spatialdecon import SpatialDecon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)

# TODO: make this a property of the dataset class?
DATASETS = ["CARD_synthetic", "GSE174746", "SPOTLight_synthetic"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="CARD_synthetic", choices=DATASETS, help="Name of the dataset.")
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--max_iter", type=int, default=10000, help="Maximum optimization iteration.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="Optimization threshold.")
args = parser.parse_args()
pprint(vars(args))

# Load dataset
dataset = CellTypeDeconvoDatasetLite(data_id=args.dataset, data_dir=args.datadir)

sc_count = dataset.data["ref_sc_count"]
sc_profile = None
sc_annot = dataset.data["ref_sc_annot"]
init_background = None
if 'init_background' in dataset.data:
    init_background = dataset.data['init_background']

mix_count = dataset.data["mix_count"]
true_p = dataset.data["true_p"]

ct_select = sorted(set(sc_annot.cellType.unique().tolist()) & set(true_p.columns.tolist()))
print('ct_select =', f'{ct_select}')

true_p = torch.FloatTensor(true_p.loc[:, ct_select].values)
if 'ref_cell_profile' in dataset.data:
    sc_profile = dataset.data["ref_cell_profile"]
    sc_profile = sc_profile.loc[:, ct_select].values

# Initialize and train model
spaDecon = SpatialDecon(sc_count=sc_count, sc_annot=sc_annot, mix_count=mix_count, ct_varname="cellType",
                        ct_select=ct_select, sc_profile=sc_profile, bias=args.bias, init_bias=init_background,
                        device=device)

#fit model
spaDecon.fit(lr=args.lr, max_iter=args.max_iter, print_period=100)

# Predict cell-type proportions and evaluate
pred = spaDecon.predict()

#score
mse = spaDecon.score(pred.T, true_p)
print(f"mse = {mse:7.4f}")
"""To reproduce SpatialDecon benchmarks, please refer to command lines belows:

CARD synthetic
$ python spatialdecon.py --dataset CARD_synthetic --lr .01 --max_iter 2250 --bias 1

GSE174746
$ python spatialdecon.py --dataset GSE174746 --lr .0001 --max_iter 20000 --bias 1

SPOTLight synthetic
$ python spatialdecon.py --dataset SPOTLight_synthetic --lr .01 --max_iter 500 --bias 1

"""
