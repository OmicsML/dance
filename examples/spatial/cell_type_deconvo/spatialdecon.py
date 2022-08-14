import os.path as o
import sys

#use if running from dance/examples/spatial/cell_type_deconvo
root_path = o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../../.."))
sys.path.append(root_path)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.datasets.spatial import CellTypeDeconvoDatasetLite

from dance.modules.spatial.cell_type_deconvo.spatialdecon import SpatialDecon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: make this a property of the dataset class?
DATASETS = ["CARD_synthetic", "GSE174746", "SPOTLight_synthetic", "toy1", "toy2"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="toy2", choices=DATASETS, help="Name of the dataset.")
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--max_iter", type=int, default=4000, help="Maximum optimization iteration.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="Optimization threshold.")
args = parser.parse_args()
pprint(vars(args))


dataset = CellTypeDeconvoDatasetLite(data_id=args.dataset, data_dir=args.datadir)
X = np.array(dataset.data['ref_sc_count'])
Y = np.array(dataset.data['mix_count'])

#set cell-types
cell_types = np.array([np.repeat(i + 1, X.shape[1] // 3) for i in range(3)]).flatten()


def cell_topic_profile(X, groups, axis=0, method='median'):
    ids = np.unique(groups)
    if method == "median":
        X_profile = np.array([np.median(X[:, groups == ids[i]], axis=1) for i in range(len(ids))]).T
    else:
        X_profile = np.array([np.mean(X[:, groups == ids[i]], axis=1) for i in range(len(ids))]).T
    return X_profile


X_profile = cell_topic_profile(X, cell_types, 0, "mean")

y = Y[:, :5]
print(y.shape)
x = X_profile
print(x.shape)

torch.manual_seed(17)
#run deconvolution with logNormReg (multiplicative log-normal errors)
deconLNR = SpatialDecon(in_dim=x.shape[1], out_dim=y.shape[1], device=device)
deconLNR.fit(x, y, max_iter=args.max_iter, lr=args.lr)
proportion_preds = deconLNR.predict().clone().detach()
print('MSLE score:', deconLNR.score(x, y))
print('Predicted Proportions:', proportion_preds)
