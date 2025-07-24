import argparse

import numpy as np

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stagate import Stagate
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--hidden_dims", type=list, default=[512, 32], help="hidden dimensions")
    parser.add_argument("--rad_cutoff", type=int, default=150, help="")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--high_variable_genes", type=int, default=3000, help="")
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = Stagate([args.high_variable_genes] + args.hidden_dims)
        preprocessing_pipeline = model.preprocessing_pipeline(n_top_hvgs=args.high_variable_genes,
                                                              radius=args.rad_cutoff)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
        adj, y = data.get_data(return_type="default")
        x = data.data.X.A
        edge_list_array = np.vstack(np.nonzero(adj))

        # Train and evaluate model
        model = Stagate([args.high_variable_genes] + args.hidden_dims)
        score = model.fit_score((x, edge_list_array), y, epochs=args.epochs, random_state=seed)
        scores.append(score)
        print(f"ARI: {score:.4f}")
    print(f"STAGATE {args.sample_number}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
""" To reproduce Stagate on other samples, please refer to command lines belows:
NOTE: since the stagate method is unstable, you have to run at least 5 times to get
      best performance. (same with original Stagate paper)

human dorsolateral prefrontal cortex sample 151673:
$ python stagate.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python stagate.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python stagate.py --sample_number 151507
"""
