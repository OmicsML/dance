import argparse

import numpy as np

from dance.data import Data
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdeepcluster import ScDeepCluster
from dance.utils import set_seed

# for repeatability
set_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--knn", default=20, type=int,
                        help="number of nearest neighbors, used by the Louvain algorithm")
    parser.add_argument(
        "--resolution", default=.8, type=float,
        help="resolution parameter, used by the Louvain algorithm, larger value for more number of clusters")
    parser.add_argument("--select_genes", default=0, type=int, help="number of selected genes, 0 means using all genes")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--data_file", default="mouse_bladder_cell", type=str,
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
    parser.add_argument("--maxiter", default=500, type=int)
    parser.add_argument("--pretrain_epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--pretrain_lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=1., type=float, help="coefficient of clustering loss")
    parser.add_argument("--sigma", default=2.5, type=float, help="coefficient of random noise")
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.001, type=float,
                        help="tolerance for delta clustering labels to terminate training stage")
    parser.add_argument("--ae_weights", default=None, help="file to pretrained weights, None for a new pretraining")
    parser.add_argument("--ae_weight_file", default="AE_weights.pth.tar",
                        help="file name to save model weights after the pretraining stage")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.ae_weight_file = f"scdeepcluster_{args.data_file}_{args.ae_weight_file}"

    adata, labels = ClusteringDataset(args.data_dir, args.data_file).load_data()
    adata.obsm["Group"] = labels
    data = Data(adata, train_size="all")

    preprocessing_pipeline = ScDeepCluster.preprocessing_pipeline()
    preprocessing_pipeline(data)

    # inputs: x, x_raw, n_counts
    inputs, y = data.get_train_data()
    n_clusters = len(np.unique(y))
    in_dim = inputs[0].shape[1]

    # Build and train model
    model = ScDeepCluster(input_dim=in_dim, z_dim=32, encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma,
                          gamma=args.gamma, device=args.device, pretrain_path=args.ae_weights)
    model.fit(inputs, y, n_clusters=n_clusters, y_pred_init=None, lr=args.lr, batch_size=args.batch_size,
              num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol,
              pt_batch_size=args.batch_size, pt_lr=args.pretrain_lr, pt_epochs=args.pretrain_epochs)

    # Evaluate model predictions
    score = model.score(None, y)
    print(f"{score=:.4f}")
""" Reproduction information
10X PBMC:
python scdeepcluster.py --data_file 10X_PBMC

Mouse ES:
python scdeepcluster.py --data_file mouse_ES_cell

Worm Neuron:
python scdeepcluster.py --data_file worm_neuron_cell --pretrain_epochs 300

Mouse Bladder:
python scdeepcluster.py --data_file mouse_bladder_cell --pretrain_epochs 300 --sigma 2.75
"""
