import argparse

from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.celltypist import Celltypist
from dance.utils.preprocess import cell_label_to_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type_test", type=str, help="name for the cell type information for test data",
                        default="Cell_type")
    parser.add_argument("--cell_type_train", type=str, help="name for the cell type information for training data",
                        default="Cell_type")
    parser.add_argument("--check_expression", action="store_true",
                        help="whether to check the normalization of training and test data")
    parser.add_argument("--max_iter", type=int, help="Max iteration during training", default=5)
    parser.add_argument("--n_jobs", type=int, help="Number of jobs", default=10)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", type=int, nargs="+", default=[203], help="List testing training dataset ids.")
    parser.add_argument("--tissue", default="Kidney", type=str)
    parser.add_argument("--train_dataset", type=int, nargs="+", default=[4682], help="List of training dataset ids.")
    parser.add_argument("--use_SGD", action="store_true",
                        help="Training algorithm -- weather it will be stochastic gradient descent")
    args = parser.parse_args()

    dataloader = CellTypeDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset, species=args.species,
                                 tissue=args.tissue)

    adata, cell_labels, idx_to_label, train_size = dataloader.load_data()
    adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs_names)
    data = Data(adata, train_size=train_size)
    data.set_config(label_channel="cell_type")

    x_train, y_train = data.get_train_data()
    y_train = y_train.argmax(1)
    model = Celltypist()
    model.fit(x_train, y_train, check_expression=args.check_expression, n_jobs=args.n_jobs, max_iter=args.max_iter,
              use_SGD=args.use_SGD)

    x_test, y_test = data.get_test_data()
    pred_obj = model.predict(x_test, check_expression=args.check_expression)
    pred = pred_obj.predicted_labels["predicted_labels"].values
    score = model.score(pred, y_test)
    print(f"{score=}")
"""To reproduce CellTypist benchmarks, please refer to command lines belows:

Mouse Brain
$ python celltypist.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python celltypist.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python celltypist.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
