import warnings

warnings.filterwarnings("ignore")
import argparse
import random
from argparse import Namespace

import dgl
import numpy as np
import torch

from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.graphsc import *
from dance.transforms.graph_construct import make_graph
from dance.transforms.preprocess import filter_data


def pipeline(**args):
    data = ClusteringDataset(args['data_dir'], args['dataset']).load_data()
    X = data.X
    Y = data.Y
    n_clusters = len(np.unique(Y))

    genes_idx, cells_idx = filter_data(X, highly_genes=args['nb_genes'])
    X = X[cells_idx][:, genes_idx]
    Y = Y[cells_idx]

    t0 = time.time()
    graph = make_graph(X, Y, dense_dim=args['in_feats'], node_features=args['node_features'],
                       normalize_weights=args['normalize_weights'], same_edge_values=args['same_edge_values'],
                       edge_norm=args['edge_norm'])

    labels = graph.ndata["label"]
    train_ids = np.where(labels != -1)[0]
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['n_layers'])
    dataloader = dgl.dataloading.NodeDataLoader(graph, train_ids, sampler, batch_size=args['batch_size'], shuffle=True,
                                                drop_last=False, num_workers=args['num_workers'])

    t1 = time.time()

    for run in range(args['num_run']):
        t_start = time.time()
        torch.manual_seed(run)
        torch.cuda.manual_seed_all(run)
        np.random.seed(run)
        random.seed(run)

        model = GraphSC(Namespace(**args))
        model.fit(args['epochs'], dataloader, n_clusters, args['learning_rate'], cluster=["KMeans", "Leiden"])
        pred = model.predict(n_clusters, cluster=["KMeans", "Leiden"])
        #        print(f'kmeans_pred: {pred.get("kmeans_pred")}\n leiden_pred: {pred.get("leiden_pred")}')
        score = model.score(Y, n_clusters, plot=False, cluster=["KMeans", "Leiden"])
        print(f'kmeans_ari: {score.get("kmeans_ari")}, leiden_ari: {score.get("leiden_ari")}')


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-cpu', '--use_cpu', default=False, action='store_true')
parser.add_argument('-if', '--in_feats', default=50, type=int)
parser.add_argument('-bs', '--batch_size', default=128, type=int)
parser.add_argument('-nw', '--normalize_weights', default='log_per_cell', choices=['log_per_cell', 'per_cell'])
parser.add_argument('-ac', '--activation', default='relu', choices=['leaky_relu', 'relu', 'prelu', 'gelu'])
parser.add_argument('-drop', '--dropout', default=0.1, type=float)
parser.add_argument('-nf', '--node_features', default='scale', choices=['scale_by_cell', 'scale', 'none'])
parser.add_argument('-sev', '--same_edge_values', default=False, action='store_true')
parser.add_argument('-en', '--edge_norm', default=True, action='store_true')
parser.add_argument('-hr', '--hidden_relu', default=False, action='store_true')
parser.add_argument('-hbn', '--hidden_bn', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
parser.add_argument('-nl', '--n_layers', type=int, default=1, choices=[1, 2])
parser.add_argument('-agg', '--agg', default='sum', choices=['sum', 'mean'])
parser.add_argument('-hd', '--hidden_dim', type=int, default=200)
parser.add_argument('-nh', '--n_hidden', type=int, default=1, choices=[0, 1, 2])
parser.add_argument('-h1', '--hidden_1', type=int, default=300)
parser.add_argument('-h2', '--hidden_2', type=int, default=0)
parser.add_argument('-ng', '--nb_genes', type=int, default=3000)
parser.add_argument('-nr', '--num_run', type=int, default=1)
parser.add_argument('-nbw', '--num_workers', type=int, default=0)
parser.add_argument('-eve', '--eval_epoch', default=True, action='store_true')
parser.add_argument('-show', '--show_epoch_ari', default=False, action='store_true')
parser.add_argument('-plot', '--plot', default=False, action='store_true')
parser.add_argument('-dd', '--data_dir', default='./data', type=str)
parser.add_argument('-data', '--dataset', default='10X_PBMC',
                    choices=['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell'])

args = parser.parse_args()
pipeline(**vars(args))
""" Reproduction information
10X PBMC:
python graphsc.py --dataset='10X_PBMC'

Mouse ES:
python graphsc.py --dataset='mouse_ES_cell'

Worm Neuron:
python graphsc.py --dataset='worm_neuron_cell'

Mouse Bladder:
python graphsc.py --dataset='mouse_bladder_cell'
"""
