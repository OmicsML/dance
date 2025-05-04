import argparse
from pprint import pprint

import numpy as np

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.card import Card
from dance.utils import set_seed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cache", action="store_true", help="Cache processed data.")
parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.AVAILABLE_DATA)
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--max_iter", type=int, default=10, help="Maximum optimization iteration.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="Optimization threshold.")
parser.add_argument("--location_free", action="store_true", help="Do not supply spatial location if set.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_runs", type=int, default=1)
args = parser.parse_args()
pprint(vars(args))

scores = []
for seed in range(args.seed, args.seed + args.num_runs):
    set_seed(seed)

    # Load dataset
    preprocessing_pipeline = Card.preprocessing_pipeline()
    dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
    data = dataset.load_data(transform=preprocessing_pipeline, cache=args.cache)

    # inputs: x_count, x_spatial
    inputs, y = data.get_data(split_name="test", return_type="numpy")
    basis = data.get_feature(return_type="default", channel="CellTopicProfile", channel_type="varm")

    # Train and evaluate model
    model = Card(basis, random_state=seed)
    score = model.fit_score(inputs, y, max_iter=args.max_iter, epsilon=args.epsilon, location_free=args.location_free)
    scores.append(score)
    print(f"MSE: {score:7.4f}")
print(f"CARD {args.dataset}:")
print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
"""To reproduce CARD benchmarks, please refer to command lines belows:

GSE174746:
$ python card.py --dataset GSE174746 --location_free

CARD synthetic:
$ python card.py --dataset CARD_synthetic

SPOTLight synthetic:
$ python card.py --dataset SPOTLight_synthetic --location_free

"""
