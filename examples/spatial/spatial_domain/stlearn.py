# Copyright 2022 DSE lab.  All rights reserved.

# TODO：start from project root directory to solve the package import error
import os
import sys

from sklearn.decomposition import PCA

sys.path.append("../../..")

import argparse

import scanpy as sc

from dance.datasets.spatial import SpotDataset
from dance.modules.spatial.spatial_domain import louvain, stlearn
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN
from dance.modules.spatial.spatial_domain.stlearn import stKmeans, stPrepare
from dance.transforms.graph_construct import construct_graph, neighbors_get_adj
from dance.transforms.preprocess import (SME_normalize, extract_feature, log1p, normalize, normalize_total, pca,
                                         prefilter_cells, prefilter_genes, prefilter_specialgenes, sc_scale, scale,
                                         set_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_number", type=str, default="151673",
        help="12 samples of human dorsolateral prefrontal cortex dataset supported in the task of spatial domain task.")
    parser.add_argument("--n_clusters", type=int, default=17, help="the number of clusters")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n_components", type=int, default=50, help="the number of components in PCA")
    parser.add_argument("--device", type=str, default='cuda', help="device for resnet extract feature")
    args = parser.parse_args()

    set_seed(args.seed)

    # get data
    dataset = SpotDataset(args.sample_number, data_dir="../../../data/spot")
    ## dataset.data has repeat name , be careful

    # build graph
    # dataset.adj = construct_graph(dataset.data.obs['x'], dataset.data.obs['y'], dataset.data.obs['x_pixel'], dataset.data.obs['y_pixel'], dataset.img, beta=49, alpha=1, histology=True)

    # preprocess data
    dataset.data.var_names_make_unique()

    # prefilter_cells(dataset.data) # this operation will change the data shape
    # prefilter_specialgenes(dataset.data)
    sc.pp.filter_genes(dataset.data, min_cells=1)
    normalize_total(dataset.data)
    log1p(dataset.data)

    extract_feature(dataset.data, image=dataset.img, n_components=args.n_components, device="cuda")
    # pca for gene expression data
    pca(dataset.data, n_components=args.n_components)

    # X_pca
    stPrepare(dataset.data)
    data_SME = dataset.data.copy()
    SME_normalize(data_SME, use_data='raw')
    data_SME.X = data_SME.obsm['raw_SME_normalized']
    sc_scale(data_SME)

    mypca = PCA(n_components=50)
    data_SME.obsm["X_pca"] = mypca.fit_transform(data_SME.X)

    # Option 1: kmeans
    # stKmeans(dataset.data, n_clusters=19)
    # model = stlearn.StKmeans(n_clusters=7)
    # stKmeans(dataset.data, n_clusters=args.n_clusters)
    # model = stlearn.StKmeans(n_clusters=args.n_clusters)
    # model.fit(dataset.data)
    # prediction = model.predict()
    # print("prediction:", prediction)
    # print(model.score(dataset.data.obs['label'].values)) # 0.1989

    # Option 2: Louvain
    data_SME.adj = neighbors_get_adj(data_SME, n_neighbors=args.n_clusters, use_rep='X_pca')
    model = stlearn.StLouvain()
    model.fit(data_SME, data_SME.adj, resolution=0.6)
    predict = model.predict()
    print(model.score(data_SME.obs['label'].values))  # 0. 31  # 0.366
""" To reproduce stlearn on other samples, please refer to command lines belows:
NOTE: since the stlearn method is unstable, you have to run multiple times to get
      best performance.

 human dorsolateral prefrontal cortex sample 151673:
python stlearn.py --n_clusters=20 --sample_number=151673  --seed=93

human dorsolateral prefrontal cortex sample 151676:
python stlearn.py --n_clusters=20 --sample_number=151676 --seed=11

human dorsolateral prefrontal cortex sample 151507:
python stlearn.py --n_clusters=20 --sample_number=151507 --seed=0
"""
