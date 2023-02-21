import argparse
from pprint import pprint

import torch

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.spotlight import SPOTlight
from dance.utils import set_seed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.DATASETS)
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--rank", type=int, default=2, help="Rank of the NMF module.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--max_iter", type=int, default=4000, help="Maximum optimization iteration.")
parser.add_argument("--seed", type=int, default=17, help="Random seed.")
parser.add_argument("--device", default="auto", help="Computation device.")
args = parser.parse_args()
set_seed(args.seed)
pprint(vars(args))

# Load dataset
dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
data = dataset.load_data()

data.set_config(feature_channel=None, feature_channel_type="X", label_channel="cell_type_portion")
# data.set_config(label_channel="cell_type_portion")
x, y = data.get_data(split_name="test", return_type="numpy")
cell_types = data.data.obsm["cell_type_portion"].columns.tolist()

ref_adata = data.get_split_data("ref")
ref_count = ref_adata.to_df()
ref_annot = ref_adata.obs

model = SPOTlight(ref_count, ref_annot, "cellType", cell_types, rank=args.rank, bias=args.bias, device=args.device)
pred = model.fit_and_predict(x, lr=args.lr, max_iter=args.max_iter)
mse = model.score(pred, torch.FloatTensor(y))

print(f"Predicted cell-type proportions of  sample 1: {pred[0].clone().cpu().numpy().round(3)}")
print(f"True cell-type proportions of  sample 1: {y[0].round(3)}")
print(f"mse = {mse:7.4f}")
"""To reproduce SpatialDecon benchmarks, please refer to command lines belows:

CARD_synthetic
$ python spotlight.py --dataset CARD_synthetic --lr .1 --max_iter 100 --rank 8 --bias 0

GSE174746
$ python spotlight.py --dataset GSE174746 --lr .1 --max_iter 15000 --rank 4 --bias 0

SPOTLight synthetic
$ python spotlight.py --dataset SPOTLight_synthetic --lr .1 --max_iter 150 --rank 10 --bias 0

"""
