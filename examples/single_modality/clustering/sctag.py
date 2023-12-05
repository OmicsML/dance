import argparse

import numpy as np

from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.sctag import ScTAG
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--dataset", default="mouse_bladder_cell", type=str,
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--w_a", default=1, type=float)
    parser.add_argument("--w_x", default=1, type=float)
    parser.add_argument("--w_d", default=0, type=float)
    parser.add_argument("--w_c", default=1, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--min_dist", default=0.5, type=float)
    parser.add_argument("--max_dist", default=20.0, type=float)
    parser.add_argument("--info_step", default=50, type=int)
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    args = parser.parse_args()
    aris = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Load data and perform necessary preprocessing
        dataloader = ClusteringDataset(args.data_dir, args.dataset)
        preprocessing_pipeline = ScTAG.preprocessing_pipeline(n_top_genes=args.highly_genes, n_components=args.pca_dim,
                                                              n_neighbors=args.k_neighbor)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        # inputs: adj, x, x_raw, n_counts
        inputs, y = data.get_train_data()
        n_clusters = len(np.unique(y))

        # Build and train model
        model = ScTAG(n_clusters=n_clusters, k=args.k, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                      dec_dim=args.dec_dim, dropout=args.dropout, device=args.device, alpha=args.alpha,
                      pretrain_path=f"sctag_{args.dataset}_pre.pkl")
        model.fit(inputs, y, epochs=args.epochs, pretrain_epochs=args.pretrain_epochs, lr=args.lr, w_a=args.w_a,
                  w_x=args.w_x, w_c=args.w_c, w_d=args.w_d, info_step=args.info_step, max_dist=args.max_dist,
                  min_dist=args.min_dist)

        # Evaluate model predictions
        score = model.score(None, y)
        print(f"{score=:.4f}")
        aris.append(score)

    print('sctag')
    print(args.dataset)
    print(f'aris: {aris}')
    print(f'aris: {np.mean(aris)} +/- {np.std(aris)}')
"""Reproduction information
10X PBMC:
python sctag.py --dataset 10X_PBMC --pretrain_epochs 100 --w_a 0.01 --w_x 3 --w_c 0.1 --dropout 0.5

Mouse ES:
python sctag.py --dataset mouse_ES_cell --pretrain_epochs 100 --w_a 0.01 --w_x 0.75 --w_c 1

Worm Neuron:
python sctag.py --dataset worm_neuron_cell --w_a 0.01 --w_x 2 --w_c 0.25 --k 1

Mouse Bladder:
python sctag.py --dataset mouse_bladder_cell --pretrain_epochs 100 --w_a 0.1 --w_x 2.5 --w_c 3
"""
