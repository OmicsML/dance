import argparse
import pprint

import torch

from dance import logger
from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.transforms.graph import PCACellFeatureGraph
from dance.typing import LOGLEVELS
from dance.utils.preprocess import cell_label_to_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--dense_dim", type=int, default=400, help="number of hidden gcn units")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--exclude_rate", type=float, default=0.005, help="exclude some cells less than this rate.")
    parser.add_argument("--hidden_dim", type=int, default=200, help="number of hidden gcn units")
    parser.add_argument("--log_level", type=str, default="INFO", choices=LOGLEVELS)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0,
                        help="the threshold to connect edges between cells and genes")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")

    # Load raw data
    dataloader = CellTypeDataset(species=args.species, tissue=args.tissue, threshold=args.threshold,
                                 exclude_rate=args.exclude_rate, test_dataset=args.test_dataset,
                                 train_dataset=args.train_dataset)
    adata, cell_labels, idx_to_label, train_size = dataloader.load_data()

    # Combine into dance data object
    adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs.index)
    data = Data(adata, train_size=train_size)

    # Data preprocessing
    PCACellFeatureGraph(n_components=args.dense_dim, split_name="train", log_level="INFO")(data)
    data.set_config(label_channel="cell_type")

    # Obtain training and testing data
    y_train = data.get_y(split_name="train", return_type="torch").argmax(1)
    y_test = data.get_y(split_name="test", return_type="torch")
    num_labels = y_test.shape[1]

    # Construct cell feature graph for scDeepSort
    # TODO: make api for the following block?
    g = data.data.uns["CellFeatureGraph"]
    num_genes = data.num_features
    gene_ids = torch.arange(num_genes)
    train_cell_ids = torch.LongTensor(data.train_idx) + num_genes
    test_cell_ids = torch.LongTensor(data.test_idx) + num_genes
    g_train = g.subgraph(torch.concat((gene_ids, train_cell_ids)))
    g_test = g.subgraph(torch.concat((gene_ids, test_cell_ids)))

    # Train and evaluate the model
    model = ScDeepSort(args.dense_dim, num_labels, args.hidden_dim, args.n_layers, args.species, args.tissue,
                       dropout=args.dropout, batch_size=args.batch_size, device=args.device)
    model.fit(g_train, y_train, epochs=args.n_epochs, lr=args.lr, weight_decay=args.weight_decay,
              val_ratio=args.test_rate)
    pred, unsure = model.predict(g_test)
    score = model.score(pred, y_test)
    print(f"{score=:.4f}")
"""To reproduce the benchmarking results, please run the following command:

Mouse Brain
$ python scdeepsort.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python scdeepsort.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python scdeepsort.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
