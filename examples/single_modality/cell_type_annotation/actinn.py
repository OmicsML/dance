import argparse
import os.path as osp

import numpy as np
import pandas as pd

from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.utils.preprocess import cell_label_to_adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Computation device.")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[100, 50, 25], help="Hidden dimensions.")
    parser.add_argument("--lambd", type=float, default=0.01, help="Regularization parameter")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--print_cost", action="store_true", help="Print cost when training")
    parser.add_argument("--runs", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[299], help="List of testing dataset ids.")
    parser.add_argument("--test_dir", default="test")
    parser.add_argument("--tissue", default="Testis")
    parser.add_argument("--train_dataset", nargs="+", default=[2216], help="List of training dataset ids.")
    parser.add_argument("--train_dir", default="train")
    args = parser.parse_args()

    # Load data
    # TODO: make directly pass species, tissue, dataset ids instead of paths, as done in the svm example
    train_data_paths = [
        osp.join(args.train_dir, args.species, f"{args.species}_{args.tissue}{i}_data.csv") for i in args.train_dataset
    ]
    train_label_paths = [
        osp.join(args.train_dir, args.species, f"{args.species}_{args.tissue}{i}_celltype.csv")
        for i in args.train_dataset
    ]
    # TODO: multiple test data
    test_data_path = osp.join(args.test_dir, args.species,
                              f"{args.species}_{args.tissue}{args.test_dataset[0]}_data.csv")
    test_label_path = osp.join(args.test_dir, args.species,
                               f"{args.species}_{args.tissue}{args.test_dataset[0]}_celltype.csv")

    dataloader = CellTypeDataset(data_type="actinn", train_set=train_data_paths, train_label=train_label_paths,
                                 test_set=test_data_path, test_label=test_label_path)

    x_adata, cell_labels, idx_to_label, train_size = dataloader.load_data()
    y_adata = cell_label_to_adata(cell_labels, idx_to_label, obs=x_adata.obs)
    data = Data(x_adata, y_adata, train_size=train_size)

    model = ACTINN(input_dim=data.num_features, output_dim=len(idx_to_label), hidden_dims=args.hidden_dims,
                   lr=args.learning_rate, device=args.device, num_epochs=args.num_epochs, batch_size=args.batch_size,
                   print_cost=args.print_cost, lambd=args.lambd)

    x_train, y_train = data.get_train_data(return_type="torch")
    x_test, y_test = data.get_test_data(return_type="torch")

    scores = []
    for k in range(args.runs):
        model.fit(x_train, y_train, seed=args.seed + k)
        pred = model.predict(x_test)
        scores.append(score := model.score(pred, y_test))
        print(f"{score}")
    print(f"Score: {np.mean(scores):04.3f} +/- {np.std(scores):04.3f}")
"""To reproduce ACTINN benchmarks, please refer to command lines belows:

Mouse Brain
$ python actinn.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695 --lambd 0.1

Mouse Spleen
$ python actinn.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --lambd 0.0001

Mouse Kidney
$ python actinn.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203 --lambd 0.01

"""
