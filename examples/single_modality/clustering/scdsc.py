import argparse
from argparse import Namespace
from time import time as get_time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

from dance.data import Data
from dance.datasets.singlemodality import *
from dance.modules.single_modality.clustering.scdsc import *
from dance.transforms.graph_construct import construct_graph_scdsc
from dance.transforms.preprocess import filter_data, normalize_adata
from dance.utils import set_seed

# for repeatability
set_seed(42)

if __name__ == "__main__":

    time_start = get_time()
    parser = argparse.ArgumentParser()

    # model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1)]
    model_para = [512, 256, 256]
    # Cluster_para = [n_z1, n_z2, n_z3, n_init, n_input, n_clusters]
    Cluster_para = [256, 128, 32, 20, 100, 10]
    # Balance_para = [binary_crossentropy_loss, ce_loss, re_loss, zinb_loss, sigma]
    Balance_para = [1, 0.01, 0.1, 0.1, 1]

    parser.add_argument(
        '--name', type=str,
        default='worm_neuron_cell')  # choice=['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell']
    parser.add_argument('--pretrain_path', type=str, default='worm_neuron_cell')
    parser.add_argument('--graph_path', type=str, default='worm_neuron_cell')
    parser.add_argument('--method', type=str, default='p')  # choice=['heat', 'cos', 'ncos', 'p']
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_enc_1', default=model_para[0], type=int)
    parser.add_argument('--n_enc_2', default=model_para[1], type=int)
    parser.add_argument('--n_enc_3', default=model_para[2], type=int)
    parser.add_argument('--n_dec_1', default=model_para[2], type=int)
    parser.add_argument('--n_dec_2', default=model_para[1], type=int)
    parser.add_argument('--n_dec_3', default=model_para[0], type=int)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--n_z1', default=Cluster_para[0], type=int)
    parser.add_argument('--n_z2', default=Cluster_para[1], type=int)
    parser.add_argument('--n_z3', default=Cluster_para[2], type=int)
    parser.add_argument('--n_input', type=int, default=Cluster_para[4])
    parser.add_argument('--n_clusters', type=int, default=Cluster_para[5])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--v', type=int, default=1)
    parser.add_argument('--nb_genes', type=int, default=2000)
    parser.add_argument('--binary_crossentropy_loss', type=float, default=Balance_para[0])
    parser.add_argument('--ce_loss', type=float, default=Balance_para[1])
    parser.add_argument('--re_loss', type=float, default=Balance_para[2])
    parser.add_argument('--zinb_loss', type=float, default=Balance_para[3])
    parser.add_argument('--sigma', type=float, default=Balance_para[4])

    args = parser.parse_args()
    args.n_input = args.nb_genes
    args.graph_path = args.name + '_' + f'{args.topk}'
    # File = [gene_expresion data file, Graph file, h5 file, pretrain_path]
    File = [
        './data/' + args.name, "./graph/" + args.graph_path + "_graph.txt", "./data/" + args.name + ".h5",
        "./model/" + args.name + "_pre.pkl"
    ]
    args.graph_path = File[1]
    args.pretrain_path = File[3]
    if not os.path.exists("./graph/"):
        os.makedirs("./graph/")
    if not os.path.exists("./model/"):
        os.makedirs("./model/")

    adata, labels = ClusteringDataset('./data', args.name).load_data()
    adata.obsm["Group"] = labels
    data = Data(adata, train_size=adata.n_obs)
    data.set_config(label_channel="Group")

    filter_data(data, highly_genes=args.nb_genes)
    normalize_adata(data, size_factors=True, normalize_input=True, logtrans_input=True)
    construct_graph_scdsc(File[1], data, args.method, args.topk)
    X, Y = data.get_train_data()
    adata = data.data

    # pretrain AE
    model = SCDSCWrapper(Namespace(**vars(args)))
    if not os.path.exists(args.pretrain_path):
        print('Pretrain:')
        dataset_pre = PretrainDataset(X)
        model.pretrain_ae(dataset_pre, args.batch_size, args.pretrain_epochs, args.pretrain_path)

    # train scDSC
    print('Train:')
    dataset = TrainingDataset(X, Y)
    X_raw = adata.raw.X
    sf = adata.obs.size_factors
    model.fit(dataset, X_raw, sf, args.graph_path, lr=args.lr, n_epochs=args.n_epochs,
              bcl=args.binary_crossentropy_loss, cl=args.ce_loss, rl=args.re_loss, zl=args.zinb_loss)
    print("Running Time：%d seconds", get_time() - time_start)

    y_pred = model.predict()
    #    print(f'Prediction: {y_pred}')
    acc, nmi, ari = model.score(Y)
    print("ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(acc, nmi, ari))
"""Reproduction information
10X PBMC:
python scdsc.py --name='10X_PBMC' --method='cos' --topk=30 --v=7 --binary_crossentropy_loss=0.75 --ce_loss=0.5 --re_loss=0.1 --zinb_loss=2.5 --sigma=0.4

Mouse Bladder:
python scdsc.py --name='mouse_bladder_cell' --method='p' --topk=50 --v=7 --binary_crossentropy_loss=2.5 --ce_loss=0.1 --re_loss=0.5 --zinb_loss=1.5 --sigma=0.6

Mouse ES:
python scdsc.py --name='mouse_ES_cell' --method='heat' --topk=50 --v=7 --binary_crossentropy_loss=0.1 --ce_loss=0.01 --re_loss=1.5 --zinb_loss=0.5 --sigma=0.1

Worm Neuron:
python scdsc.py --name='worm_neuron_cell' --method='p' --topk=20 --v=7 --binary_crossentropy_loss=2 --ce_loss=2 --re_loss=3 --zinb_loss=0.1 --sigma=0.4
"""
