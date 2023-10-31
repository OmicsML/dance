import argparse
from pprint import pprint

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.spotlight import SPOTlight
from dance.utils import set_seed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cache", action="store_true", help="Cache processed data.")
parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.AVAILABLE_DATA)
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
preprocessing_pipeline = SPOTlight.preprocessing_pipeline()
dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
data = dataset.load_data(transform=preprocessing_pipeline, cache=args.cache)
cell_types = data.data.obsm["cell_type_portion"].columns.tolist()

x, y = data.get_data(split_name="test", return_type="torch")
ref_count = data.get_feature(split_name="ref", return_type="numpy")
ref_annot = data.get_feature(split_name="ref", return_type="numpy", channel="cellType", channel_type="obs")

# Train and evaluate model
model = SPOTlight(ref_count, ref_annot, cell_types, rank=args.rank, bias=args.bias, device=args.device)
score = model.fit_score(x, y, lr=args.lr, max_iter=args.max_iter)
print(f"MSE: {score:7.4f}")
"""To reproduce SpatialDecon benchmarks, please refer to command lines belows:

CARD_synthetic $ python spotlight.py --dataset CARD_synthetic --lr .1 --max_iter 100
--rank 8 --bias 0

GSE174746 $ python spotlight.py --dataset GSE174746 --lr .1 --max_iter 15000 --rank 4
--bias 0

SPOTLight synthetic $ python spotlight.py --dataset SPOTLight_synthetic --lr .1
--max_iter 150 --rank 10 --bias 0

"""
