import argparse
import os
from time import time

import numpy as np
import torch

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
    parser.add_argument("--gamma", default=1., type=float, help="coefficient of clustering loss")
    parser.add_argument("--sigma", default=2.5, type=float, help="coefficient of random noise")
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.001, type=float,
                        help="tolerance for delta clustering labels to terminate training stage")
    parser.add_argument("--ae_weights", default=None, help="file to pretrained weights, None for a new pretraining")
    parser.add_argument("--save_dir", default="results/scDeepCluster/",
                        help="directory to save model weights during the training stage")
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

    (x, x_raw, n_counts), y = data.get_train_data()
    n_clusters = len(np.unique(y))

    model = ScDeepCluster(input_dim=x.shape[1], z_dim=32, encodeLayer=[256, 64], decodeLayer=[64, 256],
                          sigma=args.sigma, gamma=args.gamma, device=args.device)
    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X=x, X_raw=x_raw, n_counts=n_counts, batch_size=args.batch_size,
                                   epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print(f"==> loading checkpoint {args.ae_weights}")
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint["ae_state_dict"])
        else:
            print(f"==> no checkpoint found at {args.ae_weights}")
            raise ValueError
    print(f"Pretraining time: {int(time() - t0)} seconds.")

    # model training
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model.fit(X=x, X_raw=x_raw, n_counts=n_counts, n_clusters=n_clusters, init_centroid=None, y_pred_init=None, y=y,
              lr=args.lr, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval,
              tol=args.tol, save_dir=args.save_dir)

    print(f"Total time: {int(time() - t0)} seconds.")

    y_pred = model.predict()
    print(f"Prediction (first ten): {y_pred[:10]}")

    acc, nmi, ari = model.score(y)
    print("ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(acc, nmi, ari))
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
