import argparse
from pprint import pprint

import torch

from dance.data import Data
from dance.datasets.singlemodality import CellTypeDataset
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.transforms.graph import PCACellFeatureGraph
from dance.utils.preprocess import cell_label_to_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--data_type", type=str, default="scdeepsort_exp")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--filetype", default="csv", type=str, choices=["csv", "gz"], help="data file type, csv or gz")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=400, help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=200, help="number of hidden gcn units")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--threshold", type=float, default=0,
                        help="the threshold to connect edges between cells and genes")
    parser.add_argument("--num_neighbors", type=int, default=0,
                        help="number of neighbors to sample in message passing process. 0 means all neighbors")
    parser.add_argument("--exclude_rate", type=float, default=0.005, help="exclude some cells less than this rate.")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--unsure_rate", type=float, default=2.0, help="the threshold to predict unsure cell")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true")
    parser.add_argument("--test", dest="evaluate", action="store_false")
    parser.set_defaults(evaluate=True)
    parser.add_argument("--test_dir", default="test", type=str, help="test directory")
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    params = parser.parse_args()
    pprint(vars(params))

    dataloader = CellTypeDataset(data_type="scdeepsort_exp", random_seed=params.random_seed, dense_dim=params.dense_dim,
                                 test_dataset=params.test_dataset, species=params.species, tissue=params.tissue,
                                 evaluate=params.evaluate, test_dir=params.test_dir, filetype=params.filetype,
                                 threshold=params.threshold, exclude_rate=params.exclude_rate,
                                 test_rate=params.test_rate, score=True)

    adata, cell_labels, idx_to_label, train_size = dataloader.load_data()
    adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs.index)
    data = Data(adata, train_size=train_size)
    PCACellFeatureGraph(n_components=params.dense_dim, split_name="train", log_level="INFO")(data)
    data.set_config(label_channel="cell_type")

    y_train = data.get_y(split_name="train", return_type="torch").argmax(1)
    y_test = data.get_y(split_name="test", return_type="torch")
    num_labels = y_test.shape[1]

    # TODO: make api for the following blcok?
    g = data.data.uns["CellFeatureGraph"]
    num_genes = data.num_features
    gene_ids = torch.arange(num_genes)
    train_cell_ids = torch.LongTensor(data.train_idx) + num_genes
    test_cell_ids = torch.LongTensor(data.test_idx) + num_genes
    g_train = g.subgraph(torch.concat((gene_ids, train_cell_ids)))
    g_test = g.subgraph(torch.concat((gene_ids, test_cell_ids)))

    model = ScDeepSort(params.dense_dim, num_labels, params.hidden_dim, params.n_layers, params.species, params.tissue,
                       dropout=params.dropout, batch_size=params.batch_size, device=params.device)
    model.fit(g_train, y_train, epochs=params.n_epochs, lr=params.lr, weight_decay=params.weight_decay,
              val_ratio=params.test_rate)

    pred, unsure = model.predict(g_test)
    score = model.score(pred, y_test)
    print(f"{score=}")
"""To reproduce the benchmarking results, please run the following commands:

python scdeepsort.py --data_type scdeepsort --tissue Brain --test_data 2695

python scdeepsort.py --data_type scdeepsort --tissue Spleen --test_data 1759

python scdeepsort.py --data_type scdeepsort --tissue Kidney --test_data 203

"""
