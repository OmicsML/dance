import argparse

import scanpy as sc

from dance.datasets.spatial import SpotDataset
from dance.modules.spatial.spatial_domain.stagate import Stagate
from dance.transforms.graph_construct import construct_graph, stagate_construct_graph
from dance.transforms.preprocess import (log1p, normalize, normalize_total, prefilter_cells, prefilter_genes,
                                         prefilter_specialgenes, set_seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_number", type=str, default="151673",
        help="12 samples of human dorsolateral prefrontal cortex dataset supported in the task of spatial domain task.")
    parser.add_argument("--hidden_dims", type=list, default=[512, 32], help="hidden dimensions")
    parser.add_argument("--rad_cutoff", type=int, default=150, help="")
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--n_epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--high_variable_genes", type=int, default=3000, help="")
    args = parser.parse_args()

    set_seed(args.seed)
    # from dance.modules.spatial.spatial_domain.stagan import Stagate
    # get data
    dataset = SpotDataset(args.sample_number, data_dir="../../../data/spot")
    ## dataset.data has repeat name , be careful

    # preprocess data
    dataset.data.var_names_make_unique()

    sc.pp.highly_variable_genes(dataset.data, flavor="seurat_v3", n_top_genes=args.high_variable_genes)
    normalize_total(dataset.data)
    log1p(dataset.data)

    dataset.adj = stagate_construct_graph(dataset.data, rad_cutoff=args.rad_cutoff)

    hidden_dims = args.hidden_dims

    hidden_dims = [args.high_variable_genes] + hidden_dims

    model = Stagate(hidden_dims)
    model.fit(dataset.data, dataset.adj, n_epochs=args.n_epochs)
    predict = model.predict()
    curr_ARI = model.score()
    print(curr_ARI)
""" To reproduce Stagate on other samples, please refer to command lines belows:
NOTE: since the stagate method is unstable, you have to run at least 5 times to get
      best performance. (same with original Stagate paper)

human dorsolateral prefrontal cortex sample 151673:
python stagate.py --sample_number=151673 --seed=16

human dorsolateral prefrontal cortex sample 151676:
python stagate.py --sample_number=151676 --seed=2030

human dorsolateral prefrontal cortex sample 151507:
python stagate.py --sample_number=151507 --seed=2021
"""
