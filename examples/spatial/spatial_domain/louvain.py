import argparse

import numpy as np

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.louvain import Louvain
from dance.transforms.preprocess import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=202, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = Louvain(resolution=1)
        preprocessing_pipeline = model.preprocessing_pipeline(dim=args.n_components, n_neighbors=args.neighbors)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
        adj, y = data.get_data(return_type="default")

        # Train and evaluate model
        model = Louvain(resolution=1)
        score = model.fit_score(adj, y.values.ravel())
        scores.append(score)
        print(f"ARI: {score:.4f}")
    print(f"Louvain {args.sample_number}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
""" To reproduce louvain on other samples, please refer to command lines belows:
NOTE: you have to run multiple times to get best performance.

human dorsolateral prefrontal cortex sample 151673 (0.305):
$ python louvain.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676 (0.288):
$ python louvain.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507 (0.285):
$ python louvain.py --sample_number 151507
"""
