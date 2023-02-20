import argparse

import numpy as np

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stagate import Stagate
from dance.transforms.preprocess import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--hidden_dims", type=list, default=[512, 32], help="hidden dimensions")
    parser.add_argument("--rad_cutoff", type=int, default=150, help="")
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--n_epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--high_variable_genes", type=int, default=3000, help="")
    args = parser.parse_args()

    set_seed(args.seed)

    # Initialize model and get model specific preprocessing pipeline
    model = Stagate([args.high_variable_genes] + args.hidden_dims)
    preprocessing_pipeline = model.preprocessing_pipeline(n_top_hvgs=args.high_variable_genes, radius=args.rad_cutoff)

    # Load data and perform necessary preprocessing
    dataloader = SpatialLIBDDataset(data_id=args.sample_number)
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
    adj, y = data.get_data(return_type="default")

    model = Stagate([args.high_variable_genes] + args.hidden_dims)
    # TODO: extract nn model part of stagate and wrap with BaseClusteringMethod
    # TODO: extract features from adata and directly pass to model.
    model.fit(data.data, np.nonzero(adj), n_epochs=args.n_epochs)
    pred = model.predict()
    score = model.default_score_func(y.values.ravel(), pred)
    print(f"ARI: {score:.4f}")
""" To reproduce Stagate on other samples, please refer to command lines belows:
NOTE: since the stagate method is unstable, you have to run at least 5 times to get
      best performance. (same with original Stagate paper)

human dorsolateral prefrontal cortex sample 151673:
python stagate.py --sample_number=151673 --seed=16

human dorsolateral prefrontal cortex sample 151676:
python stagate.py --sample_number=151676 --seed=2030

human dorsolateral prefrontal cortex sample 151507:
python stagate.py --sample_number=151507 --seed=2021
"""
