import argparse

import numpy as np
import scanpy as sc

from dance.data import Data
from dance.datasets.spatial import SpotDataset
from dance.modules.spatial.spatial_domain.stagate import Stagate
from dance.transforms import AnnDataTransform
from dance.transforms.graph import StagateGraph
from dance.transforms.preprocess import set_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--hidden_dims", type=list, default=[512, 32], help="hidden dimensions")
    parser.add_argument("--rad_cutoff", type=int, default=150, help="")
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--n_epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--high_variable_genes", type=int, default=3000, help="")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load raw data
    dataset = SpotDataset(args.sample_number, data_dir="../../../data/spot")
    _, adata, _, spatial_pixel, label = dataset.load_data()

    # Construct dance data object
    adata.var_names_make_unique()
    adata.obsm["spatial_pixel"] = spatial_pixel
    adata.obsm["label"] = label
    data = Data(adata, train_size="all")

    # Data preprocessing pipeline
    AnnDataTransform(sc.pp.highly_variable_genes, flavor="seurat_v3", n_top_genes=args.high_variable_genes)(data)
    AnnDataTransform(sc.pp.normalize_total, target_sum=1e4)(data)
    AnnDataTransform(sc.pp.log1p)(data)

    # Construct cell graph
    StagateGraph("radius", radius=args.rad_cutoff)(data)
    data.set_config(feature_channel="StagateGraph", feature_channel_type="obsp", label_channel="label")
    adj, y = data.get_data(return_type="default")

    model = Stagate([args.high_variable_genes] + args.hidden_dims)
    model.fit(data.data, np.nonzero(adj), n_epochs=args.n_epochs)
    predict = model.predict()
    score = model.score(y.values.ravel())
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
