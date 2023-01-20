import argparse
import pprint

from dance import logger
from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.svm import SVM
from dance.transforms.cell_feature import WeightedFeaturePCA
from dance.typing import LOGLEVELS
from dance.utils.preprocess import cell_label_to_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dense_dim", type=int, default=400, help="dim of PCA")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, set to -1 for CPU")
    parser.add_argument("--log_level", type=str, default="INFO", choices=LOGLEVELS)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], type=int, help="list of dataset id")
    parser.add_argument("--tissue", default="Spleen")  # TODO: Add option for different tissue name for train/test
    parser.add_argument("--train_dataset", nargs="+", default=[1970], type=int, help="list of dataset id")

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")

    # Load raw data
    dataloader = CellTypeDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset, species=args.species,
                                 tissue=args.tissue)
    adata, cell_labels, idx_to_label, train_size = dataloader.load_data()

    # Combine into dance data object
    adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs.index)
    data = Data(adata, train_size=train_size)

    # Data preprocessing
    WeightedFeaturePCA(n_components=args.dense_dim, split_name="train", log_level="INFO")(data)
    data.set_config(feature_channel="WeightedFeaturePCA", label_channel="cell_type")

    # Obtain training and testing data
    x_train, y_train = data.get_train_data()
    y_train_converted = y_train.argmax(1)  # convert one-hot representation into label index representation
    x_test, y_test = data.get_test_data()

    # Train and evaluate the model
    model = SVM(args)
    model.fit(x_train, y_train_converted)
    pred = model.predict(x_test)
    score = model.score(pred, y_test)
    print(f"{score=:.4f}")
"""To reproduce SVM benchmarks, please refer to command lines below:

Mouse Brain
$ python svm.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python svm.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python svm.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
