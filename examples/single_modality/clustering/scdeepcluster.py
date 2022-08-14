import warnings

warnings.filterwarnings("ignore")
import os
import random
from time import time

import numpy as np
import scanpy as sc
import torch

from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdeepcluster import ScDeepCluster
from dance.transforms.preprocess import geneSelection, normalize_adata

# for repeatability
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--knn', default=20, type=int,
                        help='number of nearest neighbors, used by the Louvain algorithm')
    parser.add_argument(
        '--resolution', default=.8, type=float,
        help='resolution parameter, used by the Louvain algorithm, larger value for more number of clusters')
    parser.add_argument('--select_genes', default=0, type=int, help='number of selected genes, 0 means using all genes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--data_file', default='mouse_bladder_cell',
                        type=str)  # choice=['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell']
    parser.add_argument('--maxiter', default=500, type=int)
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--gamma', default=1., type=float, help='coefficient of clustering loss')
    parser.add_argument('--sigma', default=2.5, type=float, help='coefficient of random noise')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float,
                        help='tolerance for delta clustering labels to terminate training stage')
    parser.add_argument('--ae_weights', default=None, help='file to pretrained weights, None for a new pretraining')
    parser.add_argument('--save_dir', default='results/scDeepCluster/',
                        help='directory to save model weights during the training stage')
    parser.add_argument('--ae_weight_file', default='AE_weights.pth.tar',
                        help='file name to save model weights after the pretraining stage')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    args.ae_weight_file = f'scdeepcluster_{args.data_file}_{args.ae_weight_file}'

    data = ClusteringDataset(args.data_dir, args.data_file).load_data()
    x = data.X
    y = data.Y
    n_clusters = len(np.unique(y))

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x, dtype=np.float32)
    adata.obs['Group'] = y
    adata = adata.copy()
    adata.obs['DCA_split'] = 'train'
    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')

    adata = normalize_adata(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    input_size = adata.n_vars

    model = ScDeepCluster(input_dim=adata.n_vars, z_dim=32, encodeLayer=[256, 64], decodeLayer=[64, 256],
                          sigma=args.sigma, gamma=args.gamma, device=args.device)

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   batch_size=args.batch_size, epochs=args.pretrain_epochs,
                                   ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=n_clusters,
              init_centroid=None, y_pred_init=None, y=y, lr=args.lr, batch_size=args.batch_size,
              num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)

    print('Total time: %d seconds.' % int(time() - t0))

    y_pred = model.predict()
    #    print(f'Prediction: {y_pred}')

    acc, nmi, ari = model.score(y)
    print("ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(acc, nmi, ari))
""" Reproduction information
10X PBMC:
python scdeepcluster.py --data_file='10X_PBMC'

Mouse ES:
python scdeepcluster.py --data_file='mouse_ES_cell'

Worm Neuron:
python scdeepcluster.py --data_file='worm_neuron_cell' --pretrain_epochs 300

Mouse Bladder:
python scdeepcluster.py --data_file='mouse_bladder_cell' --pretrain_epochs 300 --sigma=2.75
"""
