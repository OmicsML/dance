import argparse

import numpy as np
import torch

from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.deepimpute import DeepImpute
from dance.utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--sub_outputdim", type=int, default=512,
                        help="Output dimension - number of genes being imputed per AE.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer dimension - number of neurons in the dense layer.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--min_cells", type=float, default=.05,
                        help="Minimum proportion of cells expressed required for a gene to pass filtering")
    parser.add_argument("--data_dir", type=str, default='data', help='test directory')
    parser.add_argument("--dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--n_top", type=int, default=5, help="Number of predictors.")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of testing set")
    parser.add_argument("--mask_rate", type=float, default=.1, help="Masking rate.")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--mask", type=bool, default=True, help="Mask data for validation.")
    params = parser.parse_args()
    print(vars(params))
    set_seed(params.random_seed)

    dataloader = ImputationDataset(data_dir=params.data_dir, dataset=params.dataset, train_size=params.train_size)
    preprocessing_pipeline = DeepImpute.preprocessing_pipeline(min_cells=params.min_cells, n_top=params.n_top,
                                                               sub_outputdim=params.sub_outputdim, mask=params.mask,
                                                               seed=params.random_seed, mask_rate=params.mask_rate)
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=params.cache)

    if params.mask:
        X, X_raw, targets, predictors, mask = data.get_x(return_type="default")
    else:
        mask = None
        X, X_raw, targets, predictors = data.get_x(return_type="default")
    X = torch.tensor(X.toarray())
    X_raw = torch.tensor(X_raw.toarray())
    train_idx = data.train_idx
    test_idx = data.test_idx

    model = DeepImpute(predictors, targets, params.dataset, params.sub_outputdim, params.hidden_dim, params.dropout,
                       params.random_seed, params.gpu)
    model.fit(X[train_idx], X[train_idx], train_idx, mask, params.batch_size, params.lr, params.n_epochs,
              params.patience)
    imputed_data = model.predict(X[test_idx], test_idx, mask)
    score = model.score(X_raw[test_idx], imputed_data, test_idx, mask, metric='RMSE')
    print("RMSE: %.4f" % score)
"""To reproduce deepimpute benchmarks, please refer to command lines belows:

Mouse Brain
$ python deepimpute.py --dataset mouse_brain_data

Mouse Embryo
$ python deepimpute.py --dataset mouse_embryo_data

PBMC
$ python graphsci.py --dataset pbmc_data

"""
