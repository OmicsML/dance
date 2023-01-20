import argparse
import pprint

from dance import logger
from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.celltypist import Celltypist
from dance.typing import LOGLEVELS
from dance.utils.preprocess import cell_label_to_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="INFO", choices=LOGLEVELS)
    parser.add_argument("--max_iter", type=int, help="Max iteration during training", default=200)
    parser.add_argument("--majority_voting", action="store_true",
                        help="Whether to refine the predicted labels via majority voting after over-clustering.")
    parser.add_argument("--n_jobs", type=int, help="Number of jobs", default=10)
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", default=[1759], help="List of testing dataset ids.")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", default=[1970], help="List of training dataset ids.")
    parser.add_argument("--not_use_SGD", action="store_true",
                        help="Training algorithm -- weather it will be stochastic gradient descent.")

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")

    # Load raw data
    dataloader = CellTypeDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset, species=args.species,
                                 tissue=args.tissue)
    adata, cell_labels, idx_to_label, train_size = dataloader.load_data()

    # Combine into dance data object
    adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs_names)
    data = Data(adata, train_size=train_size)
    data.set_config(label_channel="cell_type")

    # Obtain training and testing data
    x_train, y_train = data.get_train_data()
    y_train = y_train.argmax(1)
    x_test, y_test = data.get_test_data()

    # Train and evaluate the model
    model = Celltypist()
    model.fit(x_train, y_train, n_jobs=args.n_jobs, max_iter=args.max_iter, use_SGD=not args.not_use_SGD)
    pred = model.predict(x_test, majority_voting=args.majority_voting)
    score = model.score(pred, y_test)
    print(f"{score=:.4f}")
"""To reproduce CellTypist benchmarks, please refer to command lines below:

Mouse Brain
$ python celltypist.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python celltypist.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python celltypist.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
