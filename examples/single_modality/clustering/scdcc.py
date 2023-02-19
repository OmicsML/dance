import argparse
import os

import numpy as np

from dance.data import Data
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdcc import ScDCC
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
    parser.add_argument("--pretrain_lr", default=0.001, type=float)
    parser.add_argument("--sigma", default=2.5, type=float, help="coefficient of Gaussian noise")
    parser.add_argument("--gamma", default=1., type=float, help="coefficient of clustering loss")
    parser.add_argument("--ml_weight", default=1., type=float, help="coefficient of must-link loss")
    parser.add_argument("--cl_weight", default=1., type=float, help="coefficient of cannot-link loss")
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.00001, type=float)
    parser.add_argument("--ae_weights", default=None)
    parser.add_argument("--ae_weight_file", default="AE_weights.pth.tar")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.ae_weight_file = f"scdcc_{args.data_file}_{args.ae_weight_file}"

    adata, labels = ClusteringDataset(args.data_dir, args.data_file).load_data()
    adata.obsm["Group"] = labels
    data = Data(adata, train_size="all")

    preprocessing_pipeline = ScDCC.preprocessing_pipeline()
    preprocessing_pipeline(data)

    # inputs: x, x_raw, n_counts
    inputs, y = data.get_train_data()
    n_clusters = len(np.unique(y))
    in_dim = inputs[0].shape[1]

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

    # Build and train moodel
    model = ScDCC(input_dim=in_dim, z_dim=32, n_clusters=n_clusters, encodeLayer=[256, 64], decodeLayer=[64, 256],
                  sigma=args.sigma, gamma=args.gamma, ml_weight=args.ml_weight, cl_weight=args.ml_weight,
                  device=args.device, pretrain_path=args.ae_weights)
    model.fit(inputs, y, lr=args.lr, batch_size=args.batch_size, num_epochs=args.maxiter, ml_ind1=ml_ind1,
              ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2, update_interval=args.update_interval, tol=args.tol,
              pt_batch_size=args.batch_size, pt_lr=args.pretrain_lr, pt_epochs=args.pretrain_epochs)

    # Evaluate model predictions
    score = model.score(None, y)
    print(f"{score=:.4f}")
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
