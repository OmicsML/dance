import argparse
import pprint
from typing import get_args

import numpy as np

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.singlecellnet import SingleCellNet
from dance.typing import LogLevel
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument(
        "--normalize", action="store_true", help="Whether to perform the normalization for SingleCellNet. "
        "Disabled by default since the CellTypeAnnotation data is already normalized")
    parser.add_argument("--num_rand", type=int, default=100)
    parser.add_argument("--num_top_gene_pairs", type=int, default=250)
    parser.add_argument("--num_top_genes", type=int, default=100)
    parser.add_argument("--num_trees", type=int, default=1000)
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--stratify", type=bool, default=True)
    parser.add_argument("--test_dataset", type=int, nargs="+", default=[1759],
                        help="List testing training dataset ids.")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", type=int, nargs="+", default=[1970], help="List of training dataset ids.")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=1)

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SingleCellNet(num_trees=args.num_trees)
        preprocessing_pipeline = model.preprocessing_pipeline(normalize=args.normalize,
                                                              num_top_genes=args.num_top_genes,
                                                              num_top_gene_pairs=args.num_top_gene_pairs)

        # Load data and perform necessary preprocessing
        dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        # Obtain training and testing data
        x_train, y_train = data.get_train_data(return_type="numpy")
        y_train = y_train.argmax(1)  # convert one-hot representation into label index representation
        x_test, y_test = data.get_test_data(return_type="numpy")

        # XXX: last column for 'unsure' label by the model
        # TODO: add option to base model score function to account for unsure
        y_test = np.hstack([y_test, np.zeros((y_test.shape[0], 1))])

        # Train and evaluate the model
        model.fit(x_train, y_train, stratify=args.stratify, num_rand=args.num_rand, random_state=args.seed)
        score = model.score(x_test, y_test)
        scores.append(score)
        print(f"{score=:.4f}")
    print(f"SingleCellNet {args.species} {args.tissue} {args.test_dataset}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
"""To reproduce SingleCellNet benchmarks, please refer to command lines below:

Mouse Brain
$ python singlecellnet.py --species mouse --tissue Brain --train_dataset 753 --test_dataset 2695

Mouse Spleen
$ python singlecellnet.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python singlecellnet.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
