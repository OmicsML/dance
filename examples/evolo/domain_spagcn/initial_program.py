import argparse

import numpy as np
import scanpy as sc

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN, refine
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA
from dance.transforms.filter import FilterGenesMatch
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed,sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func
from dance.typing import Sequence,Optional
import numba
import numpy as np
import torch

# EVOLVE-BLOCK-START
@numba.njit("f4(f4[:], f4[:])")
def euclidean_distance(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i])**2
    return np.sqrt(sum)


@numba.njit("f4(f4[:], f4[:])")
def pearson_distance(a, b):
    a_avg = np.sum(a) / len(a)
    b_avg = np.sum(b) / len(b)
    cov_ab1 = [x - a_avg for x in a]
    cov_ab2 = [y - b_avg for y in b]
    cov_ab = np.sum(np.array([cov_ab1[i] * cov_ab2[i] for i in range(len(cov_ab1))]))
    sq = (np.sum(np.array([(x - a_avg)**2 for x in a])) * np.sum(np.array([(x - b_avg)**2 for x in b])))**0.5
    corr_factor = cov_ab / sq
    return 1 - corr_factor  # best correlation: 0, no correlation: 1, best anti correlation: 2


@numba.njit("f4[:](f4[:])")
def mean_rank_data(x):
    """Rank data and take mean rank for ties.

    See
    https://github.com/scipy/scipy/blob/5e4a5e3785f79dd4e8930eed883da89958860db2/scipy/stats/_stats_py.py#L10123

    """
    sorter = np.argsort(x, kind="quicksort")
    inv = np.empty(sorter.size, dtype=np.intp)
    for i, j in enumerate(sorter):
        inv[j] = i

    arr = x[sorter]
    obs = np.concatenate((np.array([True]), arr[1:] != arr[:-1]))
    dense = obs.cumsum()[inv]

    count = np.concatenate((np.nonzero(obs)[0].astype(np.float32), np.array([obs.size])))
    res = np.empty(obs.size, dtype=np.float32)
    for i in range(res.size):
        res[i] = (count[dense[i]] + count[dense[i] - 1] + 1) / 2
    return res


@numba.njit("f4(f4[:], f4[:])")
def spearman_distance(x, y):
    """The Spearman rank correlation is used to evaluate if the relationship between two
    variables, X and Y is monotonic.

    The rank correlation measures how closely related the ordering of one variable to
    the other variable, with no regard to the actual values of the variables.

    """
    if len(x) != len(y):
        raise ValueError(f'X length {len(x)} does not match Y length {len(y)}')
    x_ranks = mean_rank_data(x)
    y_ranks = mean_rank_data(y)
    return pearson_distance(x_ranks, y_ranks)  # best correlation: 0, no correlation: 1, best anti correlation: 2


DIST_FUNC_ID = ["euclidean_distance", "pearson_distance", "spearman_distance"]
# XXX: parallel produce segfalt on M-chip Mac
@numba.njit("f4[:,:](f4[:,:], u4)", parallel=True, nogil=True)
def pairwise_distance(x, dist_func_id=0):
    if dist_func_id == 0:  # Euclidean distance
        dist = euclidean_distance
    elif dist_func_id == 1:
        dist = pearson_distance
    elif dist_func_id == 2:
        dist = spearman_distance
    else:
        raise ValueError("Unknown distance function ID")

    n = x.shape[0]
    mat = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            mat[i][j] = dist(x[i], x[j])
    return mat

@register_preprocessor("graph", "spatial", overwrite=True)
class SpaGCNGraph(BaseTransform):

    _DISPLAY_ATTRS = ("alpha", "beta")

    def __init__(self, alpha, beta, *, channels: Sequence[str] = ("spatial", "spatial_pixel", "image"),
                 channel_types: Sequence[str] = ("obsm", "obsm", "uns"), **kwargs):
        """Initialize SpaGCNGraph.

        Parameters
        ----------
        alpha
            Controls the color scale.
        beta
            Controls the range of the neighborhood when calculating grey values for one spot.

        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.channels = channels
        self.channel_types = channel_types

    def __call__(self, data):
        xy = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])
        img = data.get_feature(return_type="numpy", channel=self.channels[2], channel_type=self.channel_types[2])
        self.logger.info("Start calculating the adjacency matrix using the histology image")
        g = np.zeros((xy.shape[0], 3))
        beta_half = round(self.beta / 2)
        x_lim, y_lim = img.shape[:2]
        for i, (x_pixel, y_pixel) in enumerate(xy_pixel):
            top = max(0, x_pixel - beta_half)
            left = max(0, y_pixel - beta_half)
            bottom = min(x_lim, x_pixel + beta_half + 1)
            right = min(y_lim, y_pixel + beta_half + 1)
            local_view = img[top:bottom, left:right]
            g[i] = np.mean(local_view, axis=(0, 1))
        g_var = g.var(0)
        self.logger.info(f"Variances of c0, c1, c2 = {g_var}")

        z = (g * g_var).sum(1, keepdims=True) / g_var.sum()
        z = (z - z.mean()) / z.std()
        z *= xy.std(0).max() * self.alpha

        xyz = np.hstack((xy, z)).astype(np.float32)
        self.logger.info(f"Varirances of x, y, z = {xyz.var(0)}")
        data.data.obsp[self.out] = pairwise_distance(xyz, dist_func_id=0)

        return data


@register_preprocessor("graph", "spatial", overwrite=True)
class SpaGCNGraph2D(BaseTransform):

    def __init__(self, *, channel: str = "spatial_pixel", **kwargs):
        super().__init__(**kwargs)

        self.channel = channel

    def __call__(self, data):
        x = data.get_feature(channel=self.channel, channel_type="obsm", return_type="numpy")
        data.data.obsp[self.out] = pairwise_distance(x.astype(np.float32), dist_func_id=0)
        return data

# EVOLVE-BLOCK-END

def get_preprocessing_pipeline(alpha: float = 1, beta: int = 49, dim: int = 50, log_level: LogLevel = "INFO"):
    return Compose(
        FilterGenesMatch(prefixes=["ERCC", "MT-"]),
        AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
        AnnDataTransform(sc.pp.log1p),
        SpaGCNGraph(alpha=alpha, beta=beta),
        SpaGCNGraph2D(),
        CellPCA(n_components=dim),
        SetConfig({
            "feature_channel": ["CellPCA", "SpaGCNGraph", "SpaGCNGraph2D"],
            "feature_channel_type": ["obsm", "obsp", "obsp"],
            "label_channel": "label",
            "label_channel_type": "obs"
        }),
        log_level=log_level,
    )
    
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
    parser.add_argument("--obs_nums",type=int,default=10000)
    args = parser.parse_args()

    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SpaGCN(device=args.device)
        preprocessing_pipeline = get_preprocessing_pipeline(alpha=args.alpha, beta=args.beta)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data,args.obs_nums)
        preprocessing_pipeline(data)
        (x, adj, adj_2d), y = data.get_train_data()

        # Train and evaluate model
        l = model.search_l(args.p, adj, start=args.start, end=args.end, tol=args.tol, max_run=args.max_run)
        model.set_l(l)
        res = model.search_set_res((x, adj), l=l, target_num=args.n_clusters, start=0.4, step=args.step, tol=args.tol,
                                   lr=args.lr, epochs=args.epochs, max_run=args.max_run)

        model.fit((x, adj), init_spa=True, init="louvain", tol=args.tol, lr=args.lr, epochs=args.epochs,
                                 res=res)
        embed, pred = model.predict((x, adj), return_embed=True)
        score = model.default_score_func(y, pred)
        
        refined_pred = refine(sample_id=data.data.obs_names.tolist(), pred=pred.tolist(), dis=adj_2d, shape="hexagon")
        score_refined = model.default_score_func(y, refined_pred)
        
        silhouette_score = resolve_score_func("silhouette")
        calinski_harabasz_score = resolve_score_func("calinski_harabasz")
        davies_bouldin_score = resolve_score_func("davies_bouldin")
        inner_scores.append(calculate_unified_scores({
            "silhouette": silhouette_score(embed, refined_pred),
            "calinski_harabasz": calinski_harabasz_score(embed, refined_pred),
            "davies_bouldin": davies_bouldin_score(embed, refined_pred)
        }))
        print(f"ARI: {score:.4f}")

        scores.append(score_refined)
        print(f"ARI (refined): {score_refined:.4f}")
        print(data)
    print(f"SpaGCN {args.sample_number}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
""" To reproduce SpaGCN on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spagcn.py --sample_number 151673 --lr 0.1

human dorsolateral prefrontal cortex sample 151676:
$ python spagcn.py --sample_number 151676 --lr 0.02

human dorsolateral prefrontal cortex sample 151507:
$ python spagcn.py --sample_number 151507 --lr 0.009
"""
