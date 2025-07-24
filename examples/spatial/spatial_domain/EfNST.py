import argparse
import random

import numpy as np
from sklearn.metrics import adjusted_rand_score

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.EfNST import EfNsSTRunner
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151507",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--cnnType", type=str, default="efficientnet-b0")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain the model.")
    parser.add_argument("--pre_epochs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--Conv_type", type=str, default="ResGatedGraphConv")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information.")
    parser.add_argument("--pca_n_comps", type=int, default=200, help="Number of PCA components.")
    parser.add_argument("--distType", type=str, default="KDTree", help="Distance type.")
    parser.add_argument("--k", type=int, default=12, help="Number of neighbors.")
    parser.add_argument("--no_dim_reduction", action="store_true", help="Print detailed information.")
    parser.add_argument("--min_cells", type=int, default=3, help="Minimum number of cells.")
    parser.add_argument("--platform", type=str, default="Visium", help="Platform type.")
    args = parser.parse_args()

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(args.seed, extreme_mode=True)
        try:
            EfNST = EfNsSTRunner(
                platform=args.platform,
                pre_epochs=args.pre_epochs,  #### According to your own hardware, choose the number of training
                epochs=args.epochs,
                cnnType=args.cnnType,
                Conv_type=args.Conv_type,
                random_state=seed)
            dataloader = SpatialLIBDDataset(data_id=args.sample_number)
            data = dataloader.load_data(transform=None, cache=args.cache)
            preprocessing_pipeline = EfNsSTRunner.preprocessing_pipeline(
                data_name=args.sample_number, verbose=args.verbose, cnnType=args.cnnType, pca_n_comps=args.pca_n_comps,
                distType=args.distType, k=args.k, dim_reduction=not args.no_dim_reduction, min_cells=args.min_cells,
                platform=args.platform)
            preprocessing_pipeline(data)
            (x, adj), y = data.get_data()
            adata = data.data
            adata = EfNST.fit(adata, x, graph_dict=adj, pretrain=args.pretrain)
            n_domains = len(np.unique(y))
            adata = EfNST._get_cluster_data(adata, n_domains=n_domains, priori=True)
            y_pred = EfNST.predict(adata)
        finally:
            EfNST.delete_imgs(adata)
        score = adjusted_rand_score(y, y_pred)
        scores.append(score)
        print(f"ARI: {score:.4f}")
    print(f"EfNST {args.sample_number}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
