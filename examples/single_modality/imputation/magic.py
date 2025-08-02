import argparse

import numpy as np
import torch

from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.magic import MAGIC
from dance.utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=6)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--ka", type=int, default=10)
    parser.add_argument("--epsilon", type=int, default=1)
    parser.add_argument("--rescale", type=int, default=0)  #99
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 for cpu")
    parser.add_argument("--min_cells", type=float, default=.01,
                        help="Minimum proportion of cells expressed required for a gene to pass filtering")
    parser.add_argument("--data_dir", type=str, default='data', help='test directory')
    parser.add_argument("--dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--mask", type=bool, default=True, help="Mask data for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of training set")
    parser.add_argument("--mask_rate", type=float, default=.1, help="Masking rate.")
    params = parser.parse_args()
    print(vars(params))
    rmses = []
    for seed in range(params.seed, params.seed + params.num_runs):
        set_seed(seed)

        dataloader = ImputationDataset(data_dir=params.data_dir, dataset=params.dataset, train_size=params.train_size)
        #change
        preprocessing_pipeline = MAGIC.preprocessing_pipeline(min_cells=params.min_cells, dim=params.dim,
                                                              mask=params.mask, seed=seed, mask_rate=params.mask_rate)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=params.cache)

        if params.mask:
            X, X_raw, mask, X_pca = data.get_x(return_type="default")
        else:
            mask = None
            X, X_raw, X_pca = data.get_x(return_type="default")
        X = torch.tensor(X.toarray()).float()
        X_raw = torch.tensor(X_raw.toarray()).float()
        X_train = X * mask
        X_raw_train = X_raw * mask
        model = MAGIC(t=params.t, k=params.k, ka=params.ka, epsilon=params.epsilon, rescale=params.rescale,
                      gpu=params.gpu)

        # model.fit(X_train, X_train, mask, params.batch_size, params.lr, params.n_epochs, params.patience)
        imputed_data = model.predict(X_train, X_pca)
        score = model.score(X, imputed_data, mask, metric='RMSE')
        print("RMSE: %.4f" % score)
        rmses.append(score)

    print('deepimpute')
    print(params.dataset)
    print(f'rmses: {rmses}')
    print(f'rmses: {np.mean(rmses)} +/- {np.std(rmses)}')
"""To reproduce deepimpute benchmarks, please refer to command lines belows:

Mouse Brain
$ python magic.py --dataset mouse_brain_data

Mouse Embryo
$ python magic.py --dataset mouse_embryo_data

PBMC
$ python magic.py --dataset pbmc_data

"""
