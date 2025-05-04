import argparse
from pprint import pprint

import numpy as np

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.spatialdecon import SpatialDecon
from dance.utils import set_seed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cache", action="store_true", help="Cache processed data.")
parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.AVAILABLE_DATA)
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--max_iter", type=int, default=10000, help="Maximum optimization iteration.")
parser.add_argument("--device", default="auto", help="Computation device.")
parser.add_argument("--seed", type=int, default=17, help="Random seed.")
parser.add_argument("--num_runs", type=int, default=1)
args = parser.parse_args()
pprint(vars(args))

scores = []
for seed in range(args.seed, args.seed + args.num_runs):
    set_seed(seed)

    # Load dataset
    preprocessing_pipeline = SpatialDecon.preprocessing_pipeline()
    dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
    data = dataset.load_data(transform=preprocessing_pipeline, cache=args.cache)
    cell_types = data.data.obsm["cell_type_portion"].columns.tolist()

    x, y = data.get_data(split_name="test", return_type="torch")
    ct_profile = data.get_feature(return_type="torch", channel="CellTopicProfile", channel_type="varm")

    # Train and evaluate model
    spaDecon = SpatialDecon(ct_profile, ct_select=cell_types, bias=args.bias, device=args.device)
    score = spaDecon.fit_score(x, y, lr=args.lr, max_iter=args.max_iter, print_period=100)
    scores.append(score)
    print(f"MSE: {score:7.4f}")
print(f"SpatialDecon {args.dataset}:")
print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
"""To reproduce SpatialDecon benchmarks, please refer to command lines belows:

GSE174746:
$ python spatialdecon.py --dataset GSE174746 --lr .0001 --max_iter 20000 --bias 1

CARD synthetic:
$ python spatialdecon.py --dataset CARD_synthetic --lr .01 --max_iter 2250 --bias 1

SPOTLight synthetic:
$ python spatialdecon.py --dataset SPOTLight_synthetic --lr .01 --max_iter 500 --bias 1

"""
