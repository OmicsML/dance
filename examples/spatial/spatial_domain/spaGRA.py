import argparse

import numpy as np

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spaGRA import SpaGRA
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=202, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=7, help="Number of clusters.")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "louvain"],
                        help="Clustering method.")
    args = parser.parse_args()

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SpaGRA(k=args.k, n_epochs=args.n_epochs, method=args.method, random_seed=seed)
        preprocessing_pipeline = model.preprocessing_pipeline()

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        # Fit the model and evaluate
        model.fit(data.data)
        _,y = data.get_data(return_type="default")
        score = model.score(None, y.values)
        scores.append(score)
        print(f"ARI: {score:.4f}")

    print(f"SpaGRA {args.sample_number}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")

""" To reproduce SpaGRA on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spaGRA.py --sample_number 151673 --k 7 --method kmeans

human dorsolateral prefrontal cortex sample 151676:
$ python spaGRA.py --sample_number 151676 --k 7 --method kmeans

human dorsolateral prefrontal cortex sample 151507:
$ python spaGRA.py --sample_number 151507 --k 7 --method kmeans
"""
