import argparse

import numpy as np
import scanpy as sc

from dance.data import Data
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.graphsc import GraphSC
from dance.transforms import AnnDataTransform
from dance.transforms.graph import PCACellFeatureGraph
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-cpu", "--use_cpu", default=False, action="store_true")
    parser.add_argument("-if", "--in_feats", default=50, type=int)
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
    parser.add_argument("-nw", "--normalize_weights", default="log_per_cell", choices=["log_per_cell", "per_cell"])
    parser.add_argument("-ac", "--activation", default="relu", choices=["leaky_relu", "relu", "prelu", "gelu"])
    parser.add_argument("-drop", "--dropout", default=0.1, type=float)
    parser.add_argument("-nf", "--node_features", default="scale", choices=["scale_by_cell", "scale", "none"])
    parser.add_argument("-sev", "--same_edge_values", default=False, action="store_true")
    parser.add_argument("-en", "--edge_norm", default=True, action="store_true")
    parser.add_argument("-hr", "--hidden_relu", default=False, action="store_true")
    parser.add_argument("-hbn", "--hidden_bn", default=False, action="store_true")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-nl", "--n_layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("-agg", "--agg", default="sum", choices=["sum", "mean"])
    parser.add_argument("-hd", "--hidden_dim", type=int, default=200)
    parser.add_argument("-nh", "--n_hidden", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("-h1", "--hidden_1", type=int, default=300)
    parser.add_argument("-h2", "--hidden_2", type=int, default=0)
    parser.add_argument("-ng", "--nb_genes", type=int, default=3000)
    parser.add_argument("-nr", "--num_run", type=int, default=1)
    parser.add_argument("-nbw", "--num_workers", type=int, default=1)
    parser.add_argument("-eve", "--eval_epoch", default=True, action="store_true")
    parser.add_argument("-show", "--show_epoch_ari", default=False, action="store_true")
    parser.add_argument("-plot", "--plot", default=False, action="store_true")
    parser.add_argument("-dd", "--data_dir", default="./data", type=str)
    parser.add_argument("-data", "--dataset", default="10X_PBMC",
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
    args = parser.parse_args()

    adata, labels = ClusteringDataset(args.data_dir, args.dataset).load_data()
    adata.obsm["labels"] = labels
    data = Data(adata, train_size="all")

    # Filter data
    AnnDataTransform(sc.pp.filter_genes, min_counts=3)(data)
    AnnDataTransform(sc.pp.filter_cells, min_counts=1)(data)
    AnnDataTransform(sc.pp.normalize_total)(data)
    AnnDataTransform(sc.pp.log1p)(data)
    AnnDataTransform(sc.pp.highly_variable_genes, min_mean=0.0125, max_mean=4, flavor="cell_ranger", min_disp=0.5,
                     n_top_genes=args.nb_genes, subset=True)(data)

    # Normalize
    if args.normalize_weights == "log_per_cell":
        AnnDataTransform(sc.pp.log1p)(data)
        AnnDataTransform(sc.pp.normalize_total, target_sum=1)(data)
    elif args.normalize_weights == "per_cell":
        AnnDataTransform(sc.pp.normalize_total, target_sum=1)(data)
    elif args.normalize_weights != "none":
        raise ValueError(f"Unknown normalization option {args.normalize_weights!r}."
                         "Available options are: 'none', 'log_per_cell', 'per_cell'")

    # Construct cell-gene graph
    PCACellFeatureGraph(n_components=args.in_feats, normalize_edges=args.edge_norm, feat_norm_mode="standardize")(data)
    data.set_config(feature_channel="CellFeatureGraph", feature_channel_type="uns", label_channel="labels")
    graph, y = data.get_train_data()
    n_clusters = len(np.unique(y))

    for run in range(args.num_run):
        set_seed(run)
        model = GraphSC(args)
        model.fit(graph, args.epochs, n_clusters, args.learning_rate, cluster=["KMeans", "Leiden"])
        pred = model.predict(n_clusters, cluster=["KMeans", "Leiden"])
        print(f"kmeans_pred (first 10): {pred.get('kmeans_pred')[:10]}")
        print(f"leiden_pred (first 10): {pred.get('leiden_pred')[:10]}")
        score = model.score(y, n_clusters, plot=False, cluster=["KMeans", "Leiden"])
        print(f"kmeans_ari: {score.get('kmeans_ari')}, leiden_ari: {score.get('leiden_ari')}")
""" Reproduction information
10X PBMC:
python graphsc.py --dataset 10X_PBMC

Mouse ES:
python graphsc.py --dataset mouse_ES_cell

Worm Neuron:
python graphsc.py --dataset worm_neuron_cell

Mouse Bladder:
python graphsc.py --dataset mouse_bladder_cell
"""
