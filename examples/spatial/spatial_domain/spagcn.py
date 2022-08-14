# Copyright 2022 DSE lab.  All rights reserved.

# TODO：start from project root directory to solve the package import error

import argparse
import sys

from sklearn.metrics import adjusted_mutual_info_score

sys.path.append("../../..")

from dance.datasets.spatial import SpotDataset
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN, refine
from dance.transforms.graph_construct import construct_graph
from dance.transforms.preprocess import (log1p, normalize, prefilter_cells, prefilter_genes, prefilter_specialgenes,
                                         set_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_number", type=str, default="151673",
        help="12 samples of human dorsolateral prefrontal cortex dataset supported in the task of spatial domain task")
    parser.add_argument("--beta", type=int, default=49, help="")
    parser.add_argument("--alpha", type=int, default=1, help="")
    parser.add_argument("--seed", type=int, default=1, help="")
    parser.add_argument("--p", type=float, default=0.05,
                        help="percentage of total expression contributed by neighborhoods.")
    parser.add_argument("--l", type=float, default=0.5, help="the parameter to control percentage p.")
    parser.add_argument("--start", type=float, default=0.01, help="starting value for searching l.")
    parser.add_argument("--end", type=float, default=1000, help="ending value for searching l.")
    parser.add_argument("--tol", type=float, default=5e-3, help="tolerant value for searching l.")
    parser.add_argument("--max_run", type=int, default=200, help="max runs.")
    parser.add_argument("--max_epochs", type=int, default=200, help="max epochs.")
    parser.add_argument("--n_clusters", type=int, default=7, help="the number of clusters")
    parser.add_argument("--step", type=float, default=0.1, help="")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--r_seed", type=int, default=100, help="")
    parser.add_argument("--t_seed", type=int, default=100, help="")
    parser.add_argument("--n_seed", type=int, default=100, help="")
    args = parser.parse_args()

    # get data
    dataset = SpotDataset(args.sample_number, data_dir="../../../data/spot")
    ## dataset.data has repeat name , be careful

    # build graph
    dataset.adj = construct_graph(dataset.data.obs['x'], dataset.data.obs['y'], dataset.data.obs['x_pixel'],
                                  dataset.data.obs['y_pixel'], dataset.img, beta=args.beta, alpha=args.alpha,
                                  histology=True)

    # preprocess data
    dataset.data.var_names_make_unique()

    # prefilter_cells(dataset.data) # this operation will change the data shape
    prefilter_specialgenes(dataset.data)
    normalize(dataset.data)
    log1p(dataset.data)

    # model and pipeline
    model = SpaGCN()

    p = args.p
    l = 0.5
    l = model.search_l(p, dataset.adj, start=args.start, end=args.end, tol=args.tol, max_run=args.max_run)
    model.set_l(l)
    n_clusters = args.n_clusters
    res = model.search_set_res(dataset.data, dataset.adj, l=l, target_num=n_clusters, start=0.4, step=args.step,
                               tol=args.tol, lr=args.lr, max_epochs=args.max_epochs, r_seed=args.r_seed,
                               t_seed=args.t_seed, n_seed=args.n_seed, max_run=args.max_run)
    model.fit(dataset.data, dataset.adj, init_spa=True, init="louvain", tol=args.tol, lr=args.lr,
              max_epochs=args.max_epochs, res=res)
    predict = model.predict()
    adj_2d = construct_graph(x=dataset.data.obs['x_pixel'], y=dataset.data.obs['y_pixel'], histology=False)
    refined_pred = refine(sample_id=dataset.data.obs.index.tolist(), pred=predict[0].tolist(), dis=adj_2d,
                          shape='hexagon')
    print(model.score(dataset.data.obs['label'].values))
    print(adjusted_mutual_info_score(dataset.data.obs['label'].values, refined_pred))
""" To reproduce SpaGCN on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
python spagcn.py --sample_number=151673 --lr=0.1

human dorsolateral prefrontal cortex sample 151676:
python spagcn.py --sample_number=151676  --lr=0.02

human dorsolateral prefrontal cortex sample 151507:
python spagcn.py --sample_number=151507  --lr=0.009
"""
