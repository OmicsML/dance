import argparse

import scanpy as sc

from dance.data import Data
from dance.datasets.spatial import SpotDataset
from dance.modules.spatial.spatial_domain import louvain
from dance.transforms import AnnDataTransform, CellPCA
from dance.transforms.graph import NeighborGraph
from dance.transforms.preprocess import prefilter_specialgenes, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--seed", type=int, default=202, help="Random seed.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load raw data
    dataset = SpotDataset(args.sample_number, data_dir="../../../data/spot")
    _, adata, _, _, label = dataset.load_data()

    # Construct dance data object
    adata.var_names_make_unique()
    adata.obsm["label"] = label
    data = Data(adata, train_size="all")

    # Data preprocessing pipeline
    AnnDataTransform(prefilter_specialgenes)(data)
    AnnDataTransform(sc.pp.normalize_total, target_sum=1e4)(data)
    AnnDataTransform(sc.pp.log1p)(data)

    # Construct cell feature and spot graphs
    CellPCA(n_components=args.n_components, log_level="INFO")(data)
    NeighborGraph(n_neighbors=args.neighbors)(data)
    data.set_config(feature_channel="NeighborGraph", feature_channel_type="obsp", label_channel="label")
    adj, y = data.get_data(return_type="default")

    model = louvain.Louvain(resolution=1)
    model.fit(adj)
    prediction = model.predict()
    print(model.score(y.values.ravel()))
""" To reproduce louvain on other samples, please refer to command lines belows:
NOTE: you have to run multiple times to get best performance.

human dorsolateral prefrontal cortex sample 151673:
python louvain.py --sample_number=151673 --seed=5
# 0.305

human dorsolateral prefrontal cortex sample 151676:
python louvain.py --sample_number=151676 --seed=203
# 0.288

human dorsolateral prefrontal cortex sample 151507:
python louvain.py --sample_number=151507 --seed=10
# 0.285
"""
