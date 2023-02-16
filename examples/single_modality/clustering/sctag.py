import argparse
import os
import random

import numpy as np
import torch

from dance.data import Data
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.sctag import ScTAG

# for repeatability
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--data_file", default="mouse_bladder_cell", type=str,
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
    parser.add_argument("--pretrain_file", type=str, default="./sctag_mouse_bladder_cell_pre.pkl")
    parser.add_argument("--k_neighbor", default=15, type=int)
    parser.add_argument("--highly_genes", default=3000, type=int)
    parser.add_argument("--pca_dim", default=50, type=int)
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--latent_dim", default=15, type=int)
    parser.add_argument("--dec_dim", default=None, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--pretrain_epochs", default=200, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--W_a", default=1, type=float)
    parser.add_argument("--W_x", default=1, type=float)
    parser.add_argument("--W_d", default=0, type=float)
    parser.add_argument("--W_c", default=1, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--min_dist", default=0.5, type=float)
    parser.add_argument("--max_dist", default=20.0, type=float)
    parser.add_argument("--info_step", default=50, type=int)
    args = parser.parse_args()
    args.pretrain_file = f"sctag_{args.data_file}_pre.pkl"

    # Load data
    adata, labels = ClusteringDataset(args.data_dir, args.data_file).load_data()
    adata.obsm["Group"] = labels
    data = Data(adata, train_size="all")

    preprocessing_pipeline = ScTAG.preprocessing_pipeline(n_top_genes=args.highly_genes, n_components=args.pca_dim,
                                                          n_neighbors=args.k_neighbor)
    preprocessing_pipeline(data)

    (x, x_raw, n_counts, adj), y = data.get_train_data()
    n_clusters = len(np.unique(y))

    # Build model & training
    model = ScTAG(x, adj=adj, n_clusters=n_clusters, k=args.k, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                  dec_dim=args.dec_dim, dropout=args.dropout, device=args.device, alpha=args.alpha)

    if not os.path.exists(args.pretrain_file):
        model.pre_train(x, x_raw, args.pretrain_file, n_counts, epochs=args.pretrain_epochs, info_step=args.info_step,
                        W_a=args.W_a, W_x=args.W_x, W_d=args.W_d, min_dist=args.min_dist, max_dist=args.max_dist)
    else:
        print(f"==> loading checkpoint {args.pretrain_file}")
        checkpoint = torch.load(args.pretrain_file)
        model.load_state_dict(checkpoint)

    model.fit(x, x_raw, y, n_counts, epochs=args.epochs, lr=args.lr, W_a=args.W_a, W_x=args.W_x, W_c=args.W_c,
              info_step=args.info_step)

    y_pred = model.predict()
    print(f"Prediction (first ten): {y_pred[:10]}")

    acc, nmi, ari = model.score(y)
    print("ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(acc, nmi, ari))
    print(args)
"""Reproduction information
10X PBMC:
python sctag.py --pretrain_epochs 100 --data_file 10X_PBMC --W_a 0.01 --W_x 3 --W_c 0.1 --dropout 0.5

Mouse ES:
python sctag.py --pretrain_epochs 100 --data_file mouse_ES_cell --W_a 0.01 --W_x 0.75 --W_c 1

Worm Neuron:
python sctag.py --data_file worm_neuron_cell --W_a 0.01 --W_x 2 --W_c 0.25 --k 1

Mouse Bladder:
python sctag.py --pretrain_epochs 100 --data_file mouse_bladder_cell --W_a 0.1 --W_x 2.5 --W_c 3
"""
