import argparse

import torch

from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.deepimpute import DeepImpute
from dance.utils.misc import default_parser_processor


@default_parser_processor(name="DeepImpute")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
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
    parser.add_argument("--data_dir", type=str, default="data", help="test directory")
    parser.add_argument("--dataset", default="mouse_brain_data", type=str, help="dataset id")
    parser.add_argument("--n_top", type=int, default=5, help="Number of predictors.")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of testing set")
    parser.add_argument("--mask_rate", type=float, default=.1, help="Masking rate.")
    parser.add_argument("--mask", type=bool, default=True, help="Mask data for validation.")
    return parser


if __name__ == "__main__":
    args = parse_args()

    dataloader = ImputationDataset(data_dir=args.data_dir, dataset=args.dataset, train_size=args.train_size)
    preprocessing_pipeline = DeepImpute.preprocessing_pipeline(min_cells=args.min_cells, n_top=args.n_top,
                                                               sub_outputdim=args.sub_outputdim, mask=args.mask,
                                                               seed=args.seed, mask_rate=args.mask_rate)
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

    if args.mask:
        X, X_raw, targets, predictors, mask = data.get_x(return_type="default")
    else:
        mask = None
        X, X_raw, targets, predictors = data.get_x(return_type="default")
    X = torch.tensor(X.toarray())
    X_raw = torch.tensor(X_raw.toarray())
    train_idx = data.train_idx
    test_idx = data.test_idx

    model = DeepImpute(predictors, targets, args.dataset, args.sub_outputdim, args.hidden_dim, args.dropout, args.seed,
                       args.device)
    model.fit(X[train_idx], X[train_idx], train_idx, mask, args.batch_size, args.lr, args.n_epochs, args.patience)
    imputed_data = model.predict(X[test_idx], test_idx, mask)
    score = model.score(X_raw[test_idx], imputed_data, test_idx, mask, metric="RMSE")
    print("RMSE: %.4f" % score)
"""To reproduce deepimpute benchmarks, please refer to command lines belows:

Mouse Brain
$ python deepimpute.py --dataset mouse_brain_data

Mouse Embryo
$ python deepimpute.py --dataset mouse_embryo_data

PBMC
$ python graphsci.py --dataset pbmc_data

"""
