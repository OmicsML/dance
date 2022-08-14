import argparse

import scanpy as sc

from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.singlecellnet import SingleCellNet

sc.settings.verbosity = 3
sc.logging.print_header()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_type", type=str, default='singlecellnet')
    parser.add_argument("--example", type=bool, default=False, help="Whether this is a example based on Anndata")
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--train_dataset", default=4682, type=int, help="train id")
    parser.add_argument("--test_dataset", default=203, type=int, help="test id")
    parser.add_argument("--species", default='mouse', type=str)
    parser.add_argument("--tissue", default='Kidney', type=str)
    parser.add_argument("--train_dir", type=str, default='train')
    parser.add_argument("--test_dir", type=str, default='test')
    parser.add_argument("--proj_path", type=str, default='./')
    parser.add_argument("--map_path", type=str, default='map/mouse/')
    parser.add_argument("--nTopGenes", type=int, default=100)
    parser.add_argument("--nRand", type=int, default=100)
    parser.add_argument("--nTrees", type=int, default=100)
    parser.add_argument("--nTopGenePairs", type=int, default=250)
    parser.add_argument("--dLevel", type=str, default='Cell_type',
                        help="name for the cell type information for training data")
    parser.add_argument("--dtype", type=str, default='Cell_type',
                        help="name for the cell type information for test data")
    parser.add_argument("--limitToHVG", type=bool, default=True)
    parser.add_argument("--stratify", type=bool, default=True)
    args = parser.parse_args()

    if args.example:
        dataloader = CellTypeDataset(data_type="singlecellnet_exp")
        dataloader = dataloader.load_data()
        model = SingleCellNet()
        model.fit(dataloader.expTrain, nTopGenes=args.nTopGenes, nRand=args.nRand, nTrees=args.nTrees,
                  nTopGenePairs=args.nTopGenePairs, dLevel="cell_ontology_class", stratify=args.stratify,
                  limitToHVG=args.limitToHVG)
        adVal = model.predict(dataloader.expVal)
        correct_pred, accuracy = model.score_exp(adVal, dtype="cell_ontology_class")
        print(accuracy)

    else:

        dataloader = CellTypeDataset(random_seed=args.random_seed, data_type="singlecellnet", proj_path=args.proj_path,
                                     train_dir=args.train_dir, test_dir=args.test_dir, train_dataset=args.train_dataset,
                                     test_dataset=args.test_dataset, species=args.species, tissue=args.tissue,
                                     map_path=args.map_path)
        dataloader = dataloader.load_data()

        model = SingleCellNet()
        model.fit(dataloader.train_adata, nTopGenes=args.nTopGenes, nRand=args.nRand, nTrees=args.nTrees,
                  nTopGenePairs=args.nTopGenePairs, dLevel=args.dLevel, stratify=args.stratify,
                  limitToHVG=args.limitToHVG)

        adVal = model.predict(dataloader.test_adata)
        acc = model.score(adNew=adVal, test_dataset=dataloader.test_adata, map=dataloader.map_dict[args.test_dataset],
                          dtype=args.dtype)
        print(acc)
"""To reproduce SingleCellNet benchmarks, please refer to command lines belows:

Mouse Brain
$ python singlecellnet.py --species mouse --tissue Brain --train_dataset 753 --test_dataset 2695 --dLevel Cell_type --dtype Cell_type

Mouse Spleen
$ python singlecellnet.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --dLevel Cell_type --dtype Cell_type

Mouse Kidney
$ python singlecellnet.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203 --dLevel Cell_type --dtype Cell_type

"""
