import argparse
import os
from time import time

import numpy as np
import scanpy as sc
import torch

from dance.data import Data
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdcc import ScDCC
from dance.transforms import AnnDataTransform, SaveRaw
from dance.transforms.preprocess import generate_random_pair
from dance.utils import set_seed

# for repeatability
set_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--label_cells", default=0.1, type=float)
    parser.add_argument("--label_cells_files", default="label_mouse_ES_cell.txt")
    parser.add_argument("--n_pairwise", default=0, type=int)
    parser.add_argument("--n_pairwise_error", default=0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--data_file", default="mouse_ES_cell", type=str,
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
    parser.add_argument("--maxiter", default=500, type=int)
    parser.add_argument("--pretrain_epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--sigma", default=2.5, type=float, help="coefficient of Gaussian noise")
    parser.add_argument("--gamma", default=1., type=float, help="coefficient of clustering loss")
    parser.add_argument("--ml_weight", default=1., type=float, help="coefficient of must-link loss")
    parser.add_argument("--cl_weight", default=1., type=float, help="coefficient of cannot-link loss")
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.00001, type=float)
    parser.add_argument("--ae_weights", default=None)
    parser.add_argument("--save_dir", default="results/scdcc/")
    parser.add_argument("--ae_weight_file", default="AE_weights.pth.tar")
    args = parser.parse_args()
    args.ae_weight_file = f"scdcc_{args.data_file}_{args.ae_weight_file}"

    adata, labels = ClusteringDataset(args.data_dir, args.data_file).load_data()
    adata.obsm["Group"] = labels
    data = Data(adata, train_size="all")

    # Normalize data
    AnnDataTransform(sc.pp.filter_genes, min_counts=1)(data)
    AnnDataTransform(sc.pp.filter_cells, min_counts=1)(data)
    SaveRaw()(data)
    AnnDataTransform(sc.pp.normalize_total)(data)
    AnnDataTransform(sc.pp.log1p)(data)
    AnnDataTransform(sc.pp.scale)(data)

    data.set_config(
        feature_channel=[None, None, "n_counts"],
        feature_channel_type=["X", "raw_X", "obs"],
        label_channel="Group",
    )
    (x, x_raw, n_counts), y = data.get_train_data()
    n_clusters = len(np.unique(y))

    # Generate random pairs
    if not os.path.exists(args.label_cells_files):
        indx = np.arange(len(y))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells * len(y)))]
    else:
        label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

    if args.n_pairwise > 0:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num = generate_random_pair(y, label_cell_indx, args.n_pairwise,
                                                                             args.n_pairwise_error)
        print("Must link paris: %d" % ml_ind1.shape[0])
        print("Cannot link paris: %d" % cl_ind1.shape[0])
        print("Number of error pairs: %d" % error_num)
    else:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])

    # Construct moodel
    sigma = 2.75
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"
    model = ScDCC(input_dim=x.shape[1], z_dim=32, n_clusters=n_clusters, encodeLayer=[256, 64], decodeLayer=[64, 256],
                  sigma=args.sigma, gamma=args.gamma, ml_weight=args.ml_weight, cl_weight=args.ml_weight).to(device)

    # Pretrain model
    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=x, X_raw=x_raw, n_counts=n_counts, batch_size=args.batch_size,
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

    # Train model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model.fit(X=x, X_raw=x_raw, n_counts=n_counts, y=y, lr=args.lr, batch_size=args.batch_size, num_epochs=args.maxiter,
              ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2, update_interval=args.update_interval,
              tol=args.tol, save_dir=args.save_dir)
    print(f"Total time: {int(time() - t0)} seconds.")

    y_pred = model.predict()
    print(f"Prediction (first ten): {y_pred[:10]}")
    acc, nmi, ari = model.score(y)
    print("ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(acc, nmi, ari))
    if not os.path.exists(args.label_cells_files):
        np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")
""" Reproduction information
10X PBMC:
python scdcc.py --data_file 10X_PBMC --label_cells_files label_10X_PBMC.txt --gamma=1.5

Mouse ES:
python scdcc.py --data_file mouse_ES_cell --label_cells_files label_mouse_ES_cell.txt --gamma 1 --ml_weight 0.8 --cl_weight 0.8

Worm Neuron:
python scdcc.py --data_file worm_neuron_cell --label_cells_files label_worm_neuron_cell.txt --gamma 1 --pretrain_epochs 300

Mouse Bladder:
python scdcc.py --data_file mouse_bladder_cell --label_cells_files label_mouse_bladder_cell.txt --gamma 1.5 --pretrain_epochs 100 --sigma 3
"""
