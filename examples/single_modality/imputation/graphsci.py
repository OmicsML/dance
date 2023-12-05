import argparse

import numpy as np
import torch

from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.graphsci import GraphSCI
from dance.utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of training set")
    parser.add_argument("--le", type=float, default=1, help="parameter of expression loss")
    parser.add_argument("--la", type=float, default=1e-9, help="parameter of adjacency loss")
    parser.add_argument("--ke", type=float, default=1e2, help="parameter of KL divergence of expression")
    parser.add_argument("--ka", type=float, default=1, help="parameter of KL divergence of adjacency")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--data_dir", type=str, default='data', help='test directory')
    parser.add_argument("--save_dir", type=str, default='result', help='save directory')
    parser.add_argument("--filetype", type=str, default='h5', choices=['csv', 'gz', 'h5'],
                        help='data file type, csv, csv.gz, or h5')
    parser.add_argument("--dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for exponential LR decay.")
    parser.add_argument("--threshold", type=float, default=.3,
                        help="Lower bound for correlation between genes to determine edges in graph.")
    parser.add_argument("--mask_rate", type=float, default=.1, help="Masking rate.")
    parser.add_argument("--min_cells", type=float, default=.05,
                        help="Minimum proportion of cells expressed required for a gene to pass filtering")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--mask", type=bool, default=True, help="Mask data for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    params = parser.parse_args()
    print(vars(params))
    rmses = []
    for seed in range(params.seed, params.seed + params.num_runs):
        set_seed(seed)

        dataloader = ImputationDataset(data_dir=params.data_dir, dataset=params.dataset, train_size=params.train_size)
        preprocessing_pipeline = GraphSCI.preprocessing_pipeline(min_cells=params.min_cells, threshold=params.threshold,
                                                                 mask=params.mask, seed=seed,
                                                                 mask_rate=params.mask_rate)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=params.cache)

        device = "cpu" if params.gpu == -1 else f"cuda:{params.gpu}"
        if params.mask:
            X, X_raw, g, mask = data.get_x(return_type="default")
        else:
            mask = None
            X, X_raw, g = data.get_x(return_type="default")
        X = torch.tensor(X.toarray()).float()
        X_raw = torch.tensor(X_raw.toarray()).float()
        X_train = (X * mask).to(device)
        X_raw_train = (X_raw * mask).to(device)
        g = g.to(device)

        model = GraphSCI(num_cells=X.shape[0], num_genes=X.shape[1], dataset=params.dataset, dropout=params.dropout,
                         gpu=params.gpu, seed=seed)
        model.fit(X_train, X_raw_train, g, mask, params.le, params.la, params.ke, params.ka, params.n_epochs, params.lr,
                  params.weight_decay)
        model.load_model()
        imputed_data = model.predict(X_train, X_raw_train, g, mask)
        score = model.score(X, imputed_data, mask, metric='RMSE')
        print("RMSE: %.4f" % score)
        rmses.append(score)

    print('graphsci')
    print(params.dataset)
    print(f'rmses: {rmses}')
    print(f'rmses: {np.mean(rmses)} +/- {np.std(rmses)}')
"""To reproduce GraphSCI benchmarks, please refer to command lines belows:

Mouse Brain:
CUDA_VISIBLE_DEVICES=2 python graphsci.py --dataset mouse_brain_data

Mouse Embryo:
CUDA_VISIBLE_DEVICES=2 python graphsci.py --dataset mouse_embryo_data

PBMC
CUDA_VISIBLE_DEVICES=2 python graphsci.py --dataset pbmc_data

"""
