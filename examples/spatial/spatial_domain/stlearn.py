import argparse

import scanpy as sc

from dance.data import Data
from dance.datasets.spatial import SpotDataset
from dance.modules.spatial.spatial_domain.stlearn import StKmeans, StLouvain
from dance.transforms import AnnDataTransform, CellPCA, MorphologyFeature, SMEFeature
from dance.transforms.graph import NeighborGraph, SMEGraph
from dance.transforms.preprocess import set_seed

MODES = ["louvain", "kmeans"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--mode", type=str, default="louvain", choices=MODES)
    parser.add_argument("--n_clusters", type=int, default=17, help="the number of clusters")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n_components", type=int, default=50, help="the number of components in PCA")
    parser.add_argument("--device", type=str, default="cuda", help="device for resnet extract feature")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load raw data
    dataset = SpotDataset(args.sample_number, data_dir="../../../data/spot")
    image, adata, spatial, spatial_pixel, label = dataset.load_data()

    # Construct dance data object
    adata.var_names_make_unique()
    adata.obsm["spatial"] = spatial
    adata.obsm["spatial_pixel"] = spatial_pixel
    adata.uns["image"] = image
    adata.obsm["label"] = label
    data = Data(adata, train_size="all")

    # Data preprocessing pipeline
    AnnDataTransform(sc.pp.filter_genes, min_cells=1)(data)
    AnnDataTransform(sc.pp.normalize_total, target_sum=1e4)(data)
    AnnDataTransform(sc.pp.log1p)(data)

    # Construct spot feature graph
    MorphologyFeature(n_components=args.n_components, device=args.device)(data)
    CellPCA(n_components=args.n_components)(data)
    SMEGraph()(data)
    SMEFeature(n_components=args.n_components)(data)
    NeighborGraph(n_neighbors=args.n_clusters, n_pcs=10, channel="SMEFeature")(data)

    if args.mode == "kmeans":
        data.set_config(feature_channel="SMEFeature", feature_channel_type="obsm", label_channel="label")
        x, y = data.get_data(return_type="default")

        model = StKmeans(n_clusters=args.n_clusters)
    elif args.mode == "louvain":
        data.set_config(feature_channel="NeighborGraph", feature_channel_type="obsp", label_channel="label")
        x, y = data.get_data(return_type="default")

        model = StLouvain(resolution=0.6)
    else:
        raise ValueError(f"Unknown mode {args.mode!r}, available options are {MODES}")

    model.fit(x)
    prediction = model.predict()
    score = model.score(y.values.ravel())
    print(f"ARI: {score:.4f}")
""" To reproduce stlearn on other samples, please refer to command lines belows:
NOTE: since the stlearn method is unstable, you have to run multiple times to get
      best performance.

 human dorsolateral prefrontal cortex sample 151673:
python stlearn.py --n_clusters=20 --sample_number=151673  --seed=93

human dorsolateral prefrontal cortex sample 151676:
python stlearn.py --n_clusters=20 --sample_number=151676 --seed=11

human dorsolateral prefrontal cortex sample 151507:
python stlearn.py --n_clusters=20 --sample_number=151507 --seed=0
"""
