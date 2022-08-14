import argparse
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.celltypist import Celltypist

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--train_dataset", default=4682, type=int, help="train id")
    parser.add_argument("--test_dataset", default=203, type=int, help="test id")
    parser.add_argument("--cell_type_train", type=str, help="name for the cell type information for training data",
                        default="Cell_type")
    parser.add_argument("--cell_type_test", type=str, help="name for the cell type information for test data",
                        default="Cell_type")
    parser.add_argument("--check_expression", type=bool,
                        help="whether to check the normalization of training and test data", default=False)
    parser.add_argument("--species", default='mouse', type=str)
    parser.add_argument("--tissue", default='Kidney', type=str)
    parser.add_argument("--train_dir", type=str, default='train')
    parser.add_argument("--test_dir", type=str, default='test')
    parser.add_argument("--proj_path", type=str, default='./')
    parser.add_argument("--map_path", type=str, default='map/mouse/')
    parser.add_argument("--n_jobs", type=int, help="Number of jobs", default=10)
    parser.add_argument("--max_iter", type=int, help="Max iteration during training", default=5)
    parser.add_argument("--use_SGD", type=bool,
                        help="Training algorithm -- weather it will be stochastic gradient descent", default=True)
    parser.add_argument("--label_conversion", type=bool,
                        help="whether to convert cell type labels between training and test dataset for scoring",
                        default=False)
    args = parser.parse_args()

    dataloader = CellTypeDataset(random_seed=args.random_seed, data_type="celltypist", proj_path=args.proj_path,
                                 train_dir=args.train_dir, test_dir=args.test_dir, train_dataset=args.train_dataset,
                                 test_dataset=args.test_dataset, species=args.species, tissue=args.tissue,
                                 map_path=args.map_path)
    dataloader.load_data()

    model = Celltypist()
    model_fs = model.fit(dataloader.train_adata, labels=args.cell_type_train, check_expression=args.check_expression,
                         n_jobs=args.n_jobs, max_iter=args.max_iter, use_SGD=args.use_SGD)
    predictions = model.predict(dataloader.test_adata, check_expression=args.check_expression)
    accuracy = model.score(dataloader.test_adata, predictions=predictions, labels=args.cell_type_test,
                           map=dataloader.map_dict[args.test_dataset], label_conversion=True)
    print(accuracy)
"""To reproduce CellTypist benchmarks, please refer to command lines belows:

Mouse Brain
$ python celltypist.py --species mouse --tissue Brain --train_dataset 753 --test_dataset 2695  --label_conversion True

Mouse Spleen
$ python celltypist.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759  --label_conversion False

Mouse Kidney
$ python celltypist.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203  --label_conversion False

"""
