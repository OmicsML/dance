import argparse
import pprint

import numpy as np

from dance import logger
from dance.datasets.singlemodality import ScDeepSortDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.typing import LOGLEVELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--device", default="cpu", help="Computation device.")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[2000], help="Hidden dimensions.")
    parser.add_argument("--lambd", type=float, default=0.01, help="Regularization parameter")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_level", type=str, default="INFO", choices=LOGLEVELS)
    parser.add_argument("--nofilter", action="store_true", help="Disable filtering genes by expression summaries.")
    parser.add_argument(
        "--normalize", action="store_true", help="Whether to perform the normalization described in ACTINN. "
        "Disabled by default since the scDeepSort data is already normalized")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--print_cost", action="store_true", help="Print cost when training")
    parser.add_argument("--runs", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], help="List of testing dataset ids.")
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], help="List of training dataset ids.")

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")

    # Initialize model and get model specific preprocessing pipeline
    model = ACTINN(hidden_dims=args.hidden_dims, lambd=args.lambd, device=args.device)
    preprocessing_pipeline = model.preprocessing_pipeline(normalize=args.normalize, filter_genes=not args.nofilter)

    # Load data and perform necessary preprocessing
    dataloader = ScDeepSortDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset, tissue=args.tissue,
                                   species=args.species)
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

    # Obtain training and testing data
    x_train, y_train = data.get_train_data(return_type="torch")
    x_test, y_test = data.get_test_data(return_type="torch")

    # Train and evaluate models for several rounds
    scores = []
    for k in range(args.runs):
        model.fit(x_train, y_train, seed=args.seed + k, lr=args.learning_rate, num_epochs=args.num_epochs,
                  batch_size=args.batch_size, print_cost=args.print_cost)
        scores.append(score := model.score(x_test, y_test))
        print(f"{score=:.4f}")
    print(f"Score: {np.mean(scores):04.3f} +/- {np.std(scores):04.3f}")
"""To reproduce ACTINN benchmarks, please refer to command lines below:

Mouse Brain
$ python actinn.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python actinn.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python actinn.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
