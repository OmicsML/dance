import argparse
import random
from pprint import pprint

import numpy as np
import torch

from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.graphsci import GraphSCI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=2, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="proportion of testing set")
    parser.add_argument("--le", type=float, default=1, help="parameter of expression loss")
    parser.add_argument("--la", type=float, default=0.01, help="parameter of adjacency loss")
    parser.add_argument("--ke", type=float, default=1, help="parameter of KL divergence of expression")
    parser.add_argument("--ka", type=float, default=0.01, help="parameter of KL divergence of adjacency")
    parser.add_argument("--n_genes", type=int, default=2000, help="number of of highly-variable genes to keep")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--min_counts", type=int, default=100, help="filter for cells and genes")
    parser.add_argument("--data_dir", type=str, default='data', help='test directory')
    parser.add_argument("--save_dir", type=str, default='result', help='save directory')
    parser.add_argument("--filetype", type=str, default='h5', choices=['csv', 'gz', 'h5'],
                        help='data file type, csv, csv.gz, or h5')
    parser.add_argument("--train_dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for exponential LR decay.")
    parser.add_argument("--gene_corr", type=float, default=.3,
                        help="Lower bound for correlation between genes to determine edges in graph.")
    params = parser.parse_args()

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    dataloader = ImputationDataset(random_seed=params.random_seed, gpu=params.gpu, data_dir=params.data_dir,
                                   train_dataset=params.train_dataset, filetype=params.filetype)
    dataloader.download_all_data()
    dataloader.load_data(params, model='GraphSCI')
    dl_params = dataloader.params

    model = GraphSCI(num_cells=dl_params.train_data.shape[1], num_genes=dl_params.num_genes,
                     train_dataset=params.train_dataset, lr=params.lr, dropout=params.dropout,
                     weight_decay=params.weight_decay, n_epochs=params.n_epochs, gpu=params.gpu)
    model.fit(dl_params.train_data, dl_params.train_data_raw, dl_params.adj_train, dl_params.train_size_factors,
              dl_params.adj_norm_train, le=params.le, la=params.la, ke=params.ke, ka=params.la)
    imputed_data = model.predict(dl_params.test_data, dl_params.test_data_raw, dl_params.adj_norm_test,
                                 dl_params.adj_test, dl_params.test_size_factors)
    mse_cells, mse_genes = model.score(dl_params.test_data_raw, imputed_data, dl_params.test_idx, metric='MSE')
    score = mse_cells.mean(axis=0).item()
    print("MSE: %.4f" % score)
"""To reproduce GraphSCI benchmarks, please refer to command lines belows:

Mouse Brain:
$ python graphsci.py --train_dataset mouse_brain_data

Mouse Embryo:
$ python graphsci.py --train_dataset mouse_embryo_data
"""
