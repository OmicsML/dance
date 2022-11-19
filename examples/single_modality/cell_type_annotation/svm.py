import argparse
from pprint import pprint

from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.svm import SVM
from dance.utils.preprocess import cell_label_to_adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dense_dim", type=int, default=400, help="dim of PCA")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, set to -1 for CPU")
    parser.add_argument("--map_path", default="map")
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--save_dir", default="result", help="Result directory")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--statistics_path", default="pretrained")
    parser.add_argument("--test_dataset", nargs="+", default=[299, 300], type=int, help="list of dataset id")
    parser.add_argument("--test_dir", default="test")
    parser.add_argument("--threshold", type=float, default=0, help="threshold to connect edges between cells and genes")
    parser.add_argument("--tissue", default="Testis")  # TODO: Add option for different tissue name for train/test
    parser.add_argument("--train_dataset", nargs="+", default=[2216], type=int, help="list of dataset id")
    parser.add_argument("--train_dir", default="train")

    params = parser.parse_args()
    pprint(vars(params))

    dataloader = CellTypeDataset(data_type="svm", random_seed=params.random_seed, train_dataset=params.train_dataset,
                                 test_dataset=params.test_dataset, species=params.species, tissue=params.tissue,
                                 train_dir=params.train_dir, test_dir=params.test_dir, dense_dim=params.dense_dim,
                                 statistics_path=params.statistics_path, map_path=params.map_path,
                                 threshold=params.threshold, gpu=params.gpu)

    x_adata, cell_labels, idx_to_label, train_size = dataloader.load_data()
    y_adata = cell_label_to_adata(cell_labels, idx_to_label, obs=x_adata.obs)
    data = Data(x_adata, y_adata, train_size=train_size)

    x_train, y_train = data.get_train_data()
    y_train_converted = y_train.argmax(1)  # convert one-hot representation into label index representation
    model = SVM(params)
    model.fit(x_train, y_train_converted)

    x_test, y_test = data.get_test_data()
    pred = model.predict(x_test)
    score = model.score(pred, y_test)
    print(f"{score=}")
"""To reproduce SVM benchmarks, please refer to command lines belows:

Mouse Brain
$ python svm.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python svm.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python svm.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
