import argparse

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stlearn import StKmeans, StLouvain
from dance.transforms.preprocess import set_seed

MODES = ["louvain", "kmeans"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--mode", type=str, default="louvain", choices=MODES)
    parser.add_argument("--n_clusters", type=int, default=17, help="the number of clusters")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n_components", type=int, default=50, help="the number of components in PCA")
    parser.add_argument("--device", type=str, default="cuda", help="device for resnet extract feature")
    args = parser.parse_args()
    set_seed(args.seed)

    # Initialize model and get model specific preprocessing pipeline
    if args.mode == "kmeans":
        model = StKmeans(n_clusters=args.n_clusters)
    elif args.mode == "louvain":
        model = StLouvain(resolution=0.6)
    else:
        raise ValueError(f"Unknown mode {args.mode!r}, available options are {MODES}")
    preprocessing_pipeline = model.preprocessing_pipeline()

    # Load data and perform necessary preprocessing
    dataloader = SpatialLIBDDataset(data_id=args.sample_number)
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
    x, y = data.get_data(return_type="default")

    # Train and evaluate model
    score = model.fit_score(x, y.values.ravel())
    print(f"ARI: {score:.4f}")
""" To reproduce stlearn on other samples, please refer to command lines belows:
NOTE: since the stlearn method is unstable, you have to run multiple times to get
      best performance.

human dorsolateral prefrontal cortex sample 151673:
python stlearn.py --n_clusters=20 --sample_number=151673 --seed=93

human dorsolateral prefrontal cortex sample 151676:
python stlearn.py --n_clusters=20 --sample_number=151676 --seed=11

human dorsolateral prefrontal cortex sample 151507:
python stlearn.py --n_clusters=20 --sample_number=151507 --seed=0
"""
