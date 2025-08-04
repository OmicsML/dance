import argparse

import numpy as np

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN, refine
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--beta", type=int, default=49, help="")
    parser.add_argument("--alpha", type=int, default=1, help="")
    parser.add_argument("--p", type=float, default=0.05,
                        help="percentage of total expression contributed by neighborhoods.")
    parser.add_argument("--l", type=float, default=0.5, help="the parameter to control percentage p.")
    parser.add_argument("--start", type=float, default=0.01, help="starting value for searching l.")
    parser.add_argument("--end", type=float, default=1000, help="ending value for searching l.")
    parser.add_argument("--tol", type=float, default=5e-3, help="tolerant value for searching l.")
    parser.add_argument("--max_run", type=int, default=200, help="max runs.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--n_clusters", type=int, default=7, help="the number of clusters")
    parser.add_argument("--step", type=float, default=0.1, help="")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--device", default="cpu", help="Computation device.")
    parser.add_argument("--seed", type=int, default=100, help="")
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SpaGCN(device=args.device)
        preprocessing_pipeline = model.preprocessing_pipeline(alpha=args.alpha, beta=args.beta)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
        (x, adj, adj_2d), y = data.get_train_data()

        # Train and evaluate model
        l = model.search_l(args.p, adj, start=args.start, end=args.end, tol=args.tol, max_run=args.max_run)
        model.set_l(l)
        res = model.search_set_res((x, adj), l=l, target_num=args.n_clusters, start=0.4, step=args.step, tol=args.tol,
                                   lr=args.lr, epochs=args.epochs, max_run=args.max_run)

        pred = model.fit_predict((x, adj), init_spa=True, init="louvain", tol=args.tol, lr=args.lr, epochs=args.epochs,
                                 res=res)
        score = model.default_score_func(y, pred)
        print(f"ARI: {score:.4f}")

        refined_pred = refine(sample_id=data.data.obs_names.tolist(), pred=pred.tolist(), dis=adj_2d, shape="hexagon")
        score_refined = model.default_score_func(y, refined_pred)
        scores.append(score_refined)
        print(f"ARI (refined): {score_refined:.4f}")
    print(f"SpaGCN {args.sample_number}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
""" To reproduce SpaGCN on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spagcn.py --sample_number 151673 --lr 0.1

human dorsolateral prefrontal cortex sample 151676:
$ python spagcn.py --sample_number 151676 --lr 0.02

human dorsolateral prefrontal cortex sample 151507:
$ python spagcn.py --sample_number 151507 --lr 0.009
"""
