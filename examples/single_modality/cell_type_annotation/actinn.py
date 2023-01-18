import argparse

import numpy as np
import scanpy as sc

from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.transforms import AnnDataTransform, FilterGenesPercentile
from dance.utils.preprocess import cell_label_to_df

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

    dataloader = CellTypeDataset(data_type="svm", train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                 species=args.species, tissue=args.tissue)

    adata, cell_labels, idx_to_label, train_size = dataloader.load_data()
    adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs.index)
    data = Data(adata, train_size=train_size)
    data.set_config(label_channel="cell_type")

    AnnDataTransform(sc.pp.normalize_total, target_sum=1e4)(data)
    AnnDataTransform(sc.pp.log1p)(data)
    FilterGenesPercentile(min_val=1, max_val=99, log_level="INFO")(data)
    AnnDataTransform(sc.pp.scale)(data)
    FilterGenesPercentile(min_val=1, max_val=99, log_level="INFO")(data)

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
