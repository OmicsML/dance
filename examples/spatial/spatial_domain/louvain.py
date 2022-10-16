import argparse

from dance.datasets.spatial import SpotDataset
from dance.modules.spatial.spatial_domain import louvain
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN
from dance.transforms.graph_construct import construct_graph, neighbors_get_adj
from dance.transforms.preprocess import (log1p, normalize, pca, prefilter_cells, prefilter_genes,
                                         prefilter_specialgenes, set_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_number", type=str, default="151673",
        help="12 samples of human dorsolateral prefrontal cortex dataset supported in the task of spatial domain task")
    parser.add_argument("--beta", type=int, default=49, help="")
    parser.add_argument("--seed", type=int, default=202, help="")
    parser.add_argument("--alpha", type=int, default=10, help="")
    parser.add_argument("--n_components", type=int, default=50, help="the number of components in PCA")
    parser.add_argument("--neighbors", type=int, default=17, help="")
    args = parser.parse_args()

    set_seed(args.seed)
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

    pca(dataset.data, n_components=args.n_components)
    dataset.adj = neighbors_get_adj(dataset.data, n_neighbors=args.neighbors, use_rep='X_pca')
    model = louvain.Louvain()

    model.fit(dataset.data, dataset.adj, resolution=1)
    prediction = model.predict()
    # print("prediction:", prediction)
    print(model.score(dataset.data.obs['label'].values))  # 0.3058 # 0.33 target
""" To reproduce louvain on other samples, please refer to command lines belows:
NOTE: you have to run multiple times to get best performance.

human dorsolateral prefrontal cortex sample 151673:
python louvain.py --sample_number=151673 --seed=5
# 0.305

human dorsolateral prefrontal cortex sample 151676:
python louvain.py --sample_number=151676  --seed=203
# 0.288

human dorsolateral prefrontal cortex sample 151507:
python louvain.py --sample_number=151507 --seed=10
# 0.285
"""
