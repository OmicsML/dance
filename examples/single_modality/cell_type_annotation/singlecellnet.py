import argparse

from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.singlecellnet import SingleCellNet
from dance.utils.preprocess import cell_label_to_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dLevel", type=str, default="cell_type",
                        help="name for the cell type information for training data")
    parser.add_argument("--limitToHVG", type=bool, default=True)
    parser.add_argument("--nRand", type=int, default=100)
    parser.add_argument("--nTopGenePairs", type=int, default=250)
    parser.add_argument("--nTopGenes", type=int, default=100)
    parser.add_argument("--nTrees", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--stratify", type=bool, default=True)
    parser.add_argument("--test_dataset", type=int, nargs="+", default=[203], help="List testing training dataset ids.")
    parser.add_argument("--tissue", default="Kidney", type=str)
    parser.add_argument("--train_dataset", type=int, nargs="+", default=[4682], help="List of training dataset ids.")
    args = parser.parse_args()

    dataloader = CellTypeDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset, species=args.species,
                                 tissue=args.tissue)

    adata, cell_labels, idx_to_label, train_size = dataloader.load_data()
    adata.obsm[args.dLevel] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs.index)
    data = Data(adata, train_size=train_size)
    data.set_config(label_channel=args.dLevel)

    train_adata = data.data[data.train_idx]
    test_adata = data.data[data.test_idx]

    model = SingleCellNet()
    model.fit(train_adata, nTopGenes=args.nTopGenes, nRand=args.nRand, nTrees=args.nTrees,
              nTopGenePairs=args.nTopGenePairs, dLevel=args.dLevel, stratify=args.stratify, limitToHVG=args.limitToHVG)

    pred = model.predict(test_adata)
    true = data.get_y(split_name="test")
    score = model.score(pred, true)
    print(f"{score=}")
"""To reproduce SingleCellNet benchmarks, please refer to command lines belows:

Mouse Brain
$ python singlecellnet.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python singlecellnet.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python singlecellnet.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
