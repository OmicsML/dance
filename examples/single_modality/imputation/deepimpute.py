import argparse
import random
import tempfile

import numpy as np
import torch

from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.deepimpute import DeepImpute

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--sub_outputdim", type=int, default=512,
                        help="Output dimension - number of genes being imputed per AE.")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden layer dimension - number of neurons in the dense layer.")
    parser.add_argument("--test", dest='evaluate', action='store_false')
    parser.set_defaults(evaluate=True)
    parser.add_argument("--min_counts", type=int, default=1, help="filter for cells and genes")
    parser.add_argument("--patience", type=int, default=5, help="Number of hidden units.")
    parser.add_argument("--loss", type=str, default='wMSE', help="Type of loss for training.")
    parser.add_argument("--verbose", type=int, default=1, help="1 for verbose output, 0 otherwise.")
    parser.add_argument("--imputed_only", action='store_true', default=False,
                        help="Whether to return only imputed genes in matrix or full matrix.")
    parser.add_argument("--policy", type=str, default='restore',
                        help="Policy for prediction - restore only 0, or take max of raw and imputed.")

    parser.add_argument("--data_dir", type=str, default='data', help='test directory')
    parser.add_argument("--save_dir", type=str, default='result', help='save directory')
    parser.add_argument("--filetype", type=str, default='h5', choices=['csv', 'gz', 'h5'],
                        help='data file type, csv, csv.gz, or h5')
    parser.add_argument("--train_dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--test_dataset", default='pbmc_data', type=str, help="dataset id")
    parser.add_argument("--hiddem_dim", type=float, default=256, help="number of neurons in the dense layer")
    parser.add_argument("--minVMR", type=float, default=0.5, help="Minimum variance to mean ratio.")
    parser.add_argument("--ntop", type=int, default=5, help="Number of predictors.")
    parser.add_argument("--cell_subset", type=int, default=1, help="Cell subset.")
    parser.add_argument("--NN_lim", type=float, default=None, help="Minimum variance to mean ratio.")
    parser.add_argument("--n_pred", type=float, default=None, help="Number of predictors for covariance.")
    parser.add_argument("--mode", type=str, default='random',
                        help='Mode for setting gene targets: - progressive or random')

    # parser.add_argument("--genes_to_impute", type=)
    params = parser.parse_args()
    print(vars(params))
    params.genes_to_impute = None

    dataloader = ImputationDataset(
        random_seed=params.random_seed,
        gpu=params.gpu,
        # evaluate = params.evaluate,
        data_dir=params.data_dir,
        train_dataset='pbmc_data',
        test_dataset=params.test_dataset,
        filetype='h5')
    dataloader.download_all_data()
    # dataloader.download_pretrained_data()

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    dataloader.load_data(params, model='DeepImpute')
    dl_params = dataloader.params

    model = DeepImpute(
        dl_params,
        learning_rate=params.lr,
        batch_size=params.batch_size,
        max_epochs=params.n_epochs,
        patience=params.patience,
        gpu=params.gpu,
        loss=params.loss,
        # output_prefix=tempfile.mkdtemp(),
        sub_outputdim=params.sub_outputdim,
        hidden_dim=params.hidden_dim,
        verbose=params.verbose,
        imputed_only=params.imputed_only,
        policy=params.policy,
        seed=params.random_seed,
        architecture=[
            {
                "type": "dense",
                "neurons": params.hidden_dim,
                "activation": "relu"
            },
            {
                "type": "dropout",
                "rate": params.dropout
            },
        ])

    model.fit()
    imputed_data = model.predict()
    true_expr = dl_params.true_counts
    MSE_cells, MSE_genes = model.score(true_expr)
    print(MSE_cells.mean())
    print(MSE_genes.mean())
"""To reproduce deepimpute benchmarks, please refer to command lines belows:

Mouse Brain
$ python deepimpute.py --train_dataset 'mouse_brain_data' --filetype 'h5' --hidden_dim 200 --dropout 0.4

Mouse Embryo
$ python deepimpute.py --train_dataset 'mouse_embryo_data' --filetype 'gz' --hidden_dim 200 --dropout 0.4

"""
