import argparse

import numpy as np

from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.graphsc import GraphSC
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-dv", "--device", default="auto")
    parser.add_argument("-if", "--in_feats", default=50, type=int)
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
    parser.add_argument("-nw", "--normalize_weights", default="log_per_cell", choices=["log_per_cell", "per_cell"])
    parser.add_argument("-ac", "--activation", default="relu", choices=["leaky_relu", "relu", "prelu", "gelu"])
    parser.add_argument("-drop", "--dropout", default=0.1, type=float)
    parser.add_argument("-nf", "--node_features", default="scale", choices=["scale_by_cell", "scale", "none"])
    parser.add_argument("-sev", "--same_edge_values", default=False, action="store_true")
    parser.add_argument("-en", "--edge_norm", default=True, action="store_true")
    parser.add_argument("-hr", "--hidden_relu", default=False, action="store_true")
    parser.add_argument("-hbn", "--hidden_bn", default=False, action="store_true")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-nl", "--n_layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("-agg", "--agg", default="sum", choices=["sum", "mean"])
    parser.add_argument("-hd", "--hidden_dim", type=int, default=200)
    parser.add_argument("-nh", "--n_hidden", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("-h1", "--hidden_1", type=int, default=300)
    parser.add_argument("-h2", "--hidden_2", type=int, default=0)
    parser.add_argument("-ng", "--nb_genes", type=int, default=3000)
    parser.add_argument("-nr", "--num_run", type=int, default=1)
    parser.add_argument("-nbw", "--num_workers", type=int, default=1)
    parser.add_argument("-eve", "--eval_epoch", action="store_true")
    parser.add_argument("-show", "--show_epoch_ari", action="store_true")
    parser.add_argument("-plot", "--plot", default=False, action="store_true")
    parser.add_argument("-dd", "--data_dir", default="./data", type=str)
    parser.add_argument("-data", "--dataset", default="10X_PBMC",
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    args = parser.parse_args()
    aris = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Load data and perform necessary preprocessing
        dataloader = ClusteringDataset(args.data_dir, args.dataset)
        preprocessing_pipeline = GraphSC.preprocessing_pipeline(
            n_top_genes=args.nb_genes,
            normalize_weights=args.normalize_weights,
            n_components=args.in_feats,
            normalize_edges=args.edge_norm,
        )
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        graph, y = data.get_train_data()
        n_clusters = len(np.unique(y))

        # Evaluate model for several runs
        # for run in range(args.num_run):
        # set_seed(args.seed + run)
        model = GraphSC(agg=args.agg, activation=args.activation, in_feats=args.in_feats, n_hidden=args.n_hidden,
                        hidden_dim=args.hidden_dim, hidden_1=args.hidden_1, hidden_2=args.hidden_2,
                        dropout=args.dropout, n_layers=args.n_layers, hidden_relu=args.hidden_relu,
                        hidden_bn=args.hidden_bn, n_clusters=n_clusters, cluster_method="leiden",
                        num_workers=args.num_workers, device=args.device)
        model.fit(graph, epochs=args.epochs, lr=args.learning_rate, show_epoch_ari=args.show_epoch_ari,
                  eval_epoch=args.eval_epoch)
        score = model.score(None, y)
        print(f"{score=:.4f}")
        aris.append(score)

    print('graphsc')
    print(args.dataset)
    print(f'aris: {aris}')
    print(f'aris: {np.mean(aris)} +/- {np.std(aris)}')
""" Reproduction information
10X PBMC:
python graphsc.py --dataset 10X_PBMC --dropout 0.5

Mouse ES:
python graphsc.py --dataset mouse_ES_cell

Worm Neuron:
python graphsc.py --dataset worm_neuron_cell

Mouse Bladder:
python graphsc.py --dataset mouse_bladder_cell
"""
