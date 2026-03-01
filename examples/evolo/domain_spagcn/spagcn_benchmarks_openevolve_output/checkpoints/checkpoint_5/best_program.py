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
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import pairwise_distances
from skimage.util import view_as_windows
import warnings

@numba.jit("f4(f4[:], f4[:])", nopython=True)
def euclidean_distance(t1, t2):
    # Optimized vectorized distance calculation
    sum_sq = 0.0
    for i in range(t1.shape[0]):
        diff = t1[i] - t2[i]
        sum_sq += diff * diff
    return np.sqrt(sum_sq)


@numba.jit("f4(f4[:], f4[:])", nopython=True)
def pearson_distance(a, b):
    # Improved numerical stability for pearson distance
    n = len(a)
    if n < 2:
        return 0.0
    
    # Avoid division by zero
    a_avg = np.sum(a) / n
    b_avg = np.sum(b) / n
    
    # Calculate covariance and variances
    cov_ab = 0.0
    var_a = 0.0
    var_b = 0.0
    
    for i in range(n):
        diff_a = a[i] - a_avg
        diff_b = b[i] - b_avg
        cov_ab += diff_a * diff_b
        var_a += diff_a * diff_a
        var_b += diff_b * diff_b
    
    # Handle zero variance cases
    if var_a == 0 or var_b == 0:
        return 1.0 if var_a != var_b else 0.0
        
    sq = np.sqrt(var_a * var_b)
    if sq == 0:
        return 1.0
    
    corr_factor = cov_ab / sq
    return 1.0 - corr_factor  # best correlation: 0, no correlation: 1, best anti correlation: 2


@numba.jit("f4(f4[:], f4[:])", nopython=True)
def spearman_distance(x, y):
    """The Spearman rank correlation is used to evaluate if the relationship between two
    variables, X and Y is monotonic.

    The rank correlation measures how closely related the ordering of one variable to
    the other variable, with no regard to the actual values of the variables.

    """
    if len(x) != len(y):
        raise ValueError(f'X length {len(x)} does not match Y length {len(y)}')
    
    # Use a more efficient rank calculation
    n = len(x)
    if n < 2:
        return 0.0
    
    # Create indices and sort by values
    indices_x = np.argsort(x)
    indices_y = np.argsort(y)
    
    # Compute ranks
    ranks_x = np.empty(n, dtype=np.float32)
    ranks_y = np.empty(n, dtype=np.float32)
    
    # Handle ties properly
    current_rank = 1
    for i in range(n):
        if i > 0 and x[indices_x[i]] != x[indices_x[i-1]]:
            current_rank = i + 1
        ranks_x[indices_x[i]] = current_rank
    
    current_rank = 1
    for i in range(n):
        if i > 0 and y[indices_y[i]] != y[indices_y[i-1]]:
            current_rank = i + 1
        ranks_y[indices_y[i]] = current_rank
    
    # Calculate Pearson distance on ranks
    return pearson_distance(ranks_x, ranks_y)


def fast_pairwise_distance(X, metric='euclidean'):
    """Fast pairwise distance computation using optimized scipy functions."""
    # For large matrices, compute only the upper triangular part and make symmetric
    if X.shape[0] > 10000:
        # Use sparse k-NN approach for very large datasets
        return fast_sparse_pairwise_distance(X, metric)
    
    try:
        # Use sklearn's optimized pairwise distances
        if metric == 'euclidean':
            return pairwise_distances(X, metric='euclidean', n_jobs=-1)
        elif metric == 'pearson':
            # For Pearson, we'll use a custom approach
            return fast_pearson_distance(X)
        elif metric == 'spearman':
            return fast_spearman_distance(X)
        else:
            return pairwise_distances(X, metric=metric, n_jobs=-1)
    except Exception:
        # Fallback to basic implementation
        return ssd.squareform(ssd.pdist(X, metric=metric))


def fast_pearson_distance(X):
    """Optimized Pearson distance calculation."""
    n_samples = X.shape[0]
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute correlations using matrix operations
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    var_vector = np.diag(cov_matrix)
    
    # Handle zero variance case
    var_vector[var_vector == 0] = 1.0
    
    # Compute correlation matrix
    std_vector = np.sqrt(var_vector)
    cor_matrix = cov_matrix / np.outer(std_vector, std_vector)
    
    # Convert to distance (1 - correlation)
    dist_matrix = 1 - np.abs(cor_matrix)
    
    return dist_matrix


def fast_spearman_distance(X):
    """Optimized Spearman distance calculation."""
    n_samples = X.shape[0]
    # Rank each column
    ranked_X = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, X)
    
    # Apply same process as Pearson distance on ranked data
    X_centered = ranked_X - np.mean(ranked_X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    var_vector = np.diag(cov_matrix)
    var_vector[var_vector == 0] = 1.0
    
    std_vector = np.sqrt(var_vector)
    cor_matrix = cov_matrix / np.outer(std_vector, std_vector)
    
    dist_matrix = 1 - np.abs(cor_matrix)
    return dist_matrix


def fast_sparse_pairwise_distance(X, metric='euclidean', k=10):
    """Sparse k-NN approach for large datasets."""
    from sklearn.neighbors import kneighbors_graph
    import scipy.sparse as sp
    
    # Use k-nearest neighbors approach for memory efficiency
    if metric == 'euclidean':
        # Create sparse adjacency matrix with k nearest neighbors
        knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)
        # Convert to distance matrix with 1.0 for connected and 0.0 for unconnected
        # Note: This is just an approximation - for precise distances we'd need to compute them
        return knn_graph.toarray().astype(np.float32)
    else:
        # For non-Euclidean distances, fall back to regular computation
        return fast_pairwise_distance(X, metric)


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
        
        # Vectorized patch extraction using advanced indexing for better performance
        beta_half = round(self.beta / 2)
        x_lim, y_lim = img.shape[:2]
        
        # Get all pixel coordinates at once
        x_pixels = xy_pixel[:, 0].astype(int)
        y_pixels = xy_pixel[:, 1].astype(int)
        
        # Calculate bounds for all patches simultaneously
        tops = np.clip(x_pixels - beta_half, 0, x_lim)
        bottoms = np.clip(x_pixels + beta_half + 1, 0, x_lim)
        lefts = np.clip(y_pixels - beta_half, 0, y_lim)
        rights = np.clip(y_pixels + beta_half + 1, 0, y_lim)
        
        # Extract patches using vectorized operations
        if len(img.shape) == 3:
            # Handle RGB images - allocate space for RGB features
            g = np.zeros((xy.shape[0], 3), dtype=np.float32)
            
            # Process each patch efficiently
            for i in range(xy.shape[0]):
                patch = img[tops[i]:bottoms[i], lefts[i]:rights[i]]
                if patch.size > 0:
                    g[i] = np.mean(patch, axis=(0, 1))
                else:
                    # Fallback to global mean if patch is empty
                    g[i] = np.mean(img, axis=(0, 1))
        else:
            # Handle grayscale images
            g = np.zeros((xy.shape[0], 1), dtype=np.float32)
            for i in range(xy.shape[0]):
                patch = img[tops[i]:bottoms[i], lefts[i]:rights[i]]
                if patch.size > 0:
                    g[i] = np.mean(patch)
                else:
                    g[i] = np.mean(img)
        
        # Apply adaptive resolution based on local density for better spatial clustering
        # Calculate local density around each spot and adjust the spatial weights accordingly
        if xy.shape[0] > 1:  # Only if we have multiple spots
            # Calculate distances to nearby points to estimate local density
            from sklearn.neighbors import NearestNeighbors
            n_neighbors = min(10, xy.shape[0] - 1)  # Use fewer neighbors to avoid O(N^2) overhead
            
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(xy)
            distances, indices = nbrs.kneighbors(xy)
            
            # Local density metric (inverse of average distance to neighbors)
            local_density = 1.0 / (np.mean(distances, axis=1) + 1e-8)
            
            # Normalize density to influence the histology weight adaptively
            density_factor = local_density / (np.mean(local_density) + 1e-8)
            
            # Adjust histology features based on local density adaptively
            adaptive_g = g * density_factor.reshape(-1, 1)
            
            # Update g with adaptive features for better spatial clustering
            g = adaptive_g
        
        # Enhanced feature extraction with variance and texture features
        g_var = g.var(0)
        self.logger.info(f"Variances of c0, c1, c2 = {g_var}")
        
        # Enhanced z-coordinate calculation with numerical stability
        if len(g_var) > 0 and np.sum(g_var) > 0:
            z = (g * g_var).sum(1, keepdims=True) / g_var.sum()
        else:
            z = np.zeros((g.shape[0], 1))
            
        # Normalize z coordinate
        if z.std() > 1e-10:  # Avoid division by zero
            z = (z - z.mean()) / z.std()
        else:
            z = z - z.mean()
            
        z *= xy.std(0).max() * self.alpha

        xyz = np.hstack((xy, z)).astype(np.float32)
        self.logger.info(f"Variances of x, y, z = {xyz.var(0)}")
        
        # Use optimized distance calculation with sparse matrix for large datasets
        if xyz.shape[0] > 10000:
            # Use sparse k-NN approach for large datasets to save memory
            data.data.obsp[self.out] = fast_sparse_pairwise_distance(xyz, 'euclidean', k=min(20, xyz.shape[0]//100))
        else:
            # For smaller datasets, use full distance matrix with optimized implementation
            data.data.obsp[self.out] = fast_pairwise_distance(xyz, 'euclidean')

        return data


@register_preprocessor("graph", "spatial", overwrite=True)
class SpaGCNGraph2D(BaseTransform):

    def __init__(self, *, channel: str = "spatial_pixel", **kwargs):
        super().__init__(**kwargs)

        self.channel = channel

    def __call__(self, data):
        x = data.get_feature(channel=self.channel, channel_type="obsm", return_type="numpy")
        
        # Add numerical stability check
        if x.size == 0:
            # Create empty matrix if no data
            data.data.obsp[self.out] = np.zeros((0, 0), dtype=np.float32)
        else:
            # Use optimized distance calculation
            data.data.obsp[self.out] = fast_pairwise_distance(x.astype(np.float32), 'euclidean')
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
