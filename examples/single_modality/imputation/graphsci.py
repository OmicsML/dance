import argparse
import random
from pprint import pprint

import numpy as np
import torch

from dance.utils import set_seed
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
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    params = parser.parse_args()

    set_seed(params.random_seed)

    dataloader = ImputationDataset(data_dir=params.data_dir, train_dataset=params.train_dataset, filetype=params.filetype)
    preprocessing_pipeline = GraphSCI.preprocessing_pipeline()
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=params.cache)

    device = "cpu" if params.gpu == -1 else f"cuda:{params.gpu}"
    X, X_raw, n_counts, g = data.get_x(return_type="default")
    label, label_raw = data.get_y(return_type="torch")
    X = torch.tensor(X.toarray()).T.to(device)
    X_raw = torch.tensor(X_raw.toarray()).T.to(device)
    label = label.T.to(device)
    label_raw = label_raw.T.to(device)

    model = GraphSCI(num_cells=X.shape[1], num_genes=X.shape[0], train_dataset=params.train_dataset, 
                     dropout=params.dropout, gpu=params.gpu)
    model.fit(X, X_raw, n_counts, g, params.le, params.la, params.ke, params.la,
              params.n_epochs, params.lr, params.weight_decay)
    imputed_data = model.predict(params.test_data, params.test_data_raw, params.adj_norm_test,
                                 params.adj_test, params.test_size_factors)
    mse_cells, mse_genes = model.score(params.test_data_raw, imputed_data, params.test_idx, metric='MSE')
    score = mse_cells.mean(axis=0).item()
    print("MSE: %.4f" % score)
"""To reproduce GraphSCI benchmarks, please refer to command lines belows:

Mouse Brain:
$ python graphsci.py --train_dataset mouse_brain_data

Mouse Embryo:
$ python graphsci.py --train_dataset mouse_embryo_data
"""
