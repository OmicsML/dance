import argparse
from pprint import pprint

from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.svm import SVM

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
                                 train_dir=params.train_dir, test_dir=params.test_dir,
                                 statistics_path=params.statistics_path, map_path=params.map_path,
                                 threshold=params.threshold, gpu=params.gpu)
    dataloader = dataloader.load_data()

    model = SVM(params)
    model.fit(dataloader.svm_train_labels, dataloader.svm_train_cell_feat)
    prediction = model.predict(dataloader.svm_map_dict, dataloader.svm_id2label, dataloader.svm_test_label_dict,
                               dataloader.svm_test_feat_dict, dataloader.svm_test_cell_id_dict)
    accuracy = model.score(dataloader.svm_map_dict, dataloader.svm_id2label, dataloader.svm_test_label_dict,
                           dataloader.svm_test_feat_dict, dataloader.svm_test_cell_id_dict)
"""To reproduce SVM benchmarks, please refer to command lines belows:

Mouse Brain
$ python svm.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python svm.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python svm.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
