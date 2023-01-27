import argparse

from sklearn.metrics import adjusted_mutual_info_score

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN, refine

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
    parser.add_argument("--max_epochs", type=int, default=200, help="max epochs.")
    parser.add_argument("--n_clusters", type=int, default=7, help="the number of clusters")
    parser.add_argument("--step", type=float, default=0.1, help="")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--r_seed", type=int, default=100, help="")
    parser.add_argument("--t_seed", type=int, default=100, help="")
    parser.add_argument("--n_seed", type=int, default=100, help="")
    args = parser.parse_args()

    # Initialize model and get model specific preprocessing pipeline
    model = SpaGCN()
    preprocessing_pipeline = model.preprocessing_pipeline(alpha=args.alpha, beta=args.beta)

    # Load data and perform necessary preprocessing
    dataloader = SpatialLIBDDataset(data_id=args.sample_number)
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
    (x, adj, adj_2d), y = data.get_train_data()

    # Train and evaluate model
    l = model.search_l(args.p, adj, start=args.start, end=args.end, tol=args.tol, max_run=args.max_run)
    model.set_l(l)
    n_clusters = args.n_clusters
    res = model.search_set_res(x, adj, l=l, target_num=n_clusters, start=0.4, step=args.step, tol=args.tol, lr=args.lr,
                               max_epochs=args.max_epochs, r_seed=args.r_seed, t_seed=args.t_seed, n_seed=args.n_seed,
                               max_run=args.max_run)
    model.fit(x, adj, init_spa=True, init="louvain", tol=args.tol, lr=args.lr, max_epochs=args.max_epochs, res=res)
    predict = model.predict()

    refined_pred = refine(sample_id=data.data.obs_names.tolist(), pred=predict[0].tolist(), dis=adj_2d, shape="hexagon")
    print(model.score(y.ravel()))
    print(adjusted_mutual_info_score(y.ravel(), refined_pred))
""" To reproduce SpaGCN on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
python spagcn.py --sample_number=151673 --lr=0.1

human dorsolateral prefrontal cortex sample 151676:
python spagcn.py --sample_number=151676  --lr=0.02

human dorsolateral prefrontal cortex sample 151507:
python spagcn.py --sample_number=151507  --lr=0.009
"""
