import argparse
import time  # 新增：导入 time 模块
from typing import Optional

import numpy as np
import scanpy as sc

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.louvain import Louvain
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA
from dance.transforms.filter import FilterGenesMatch
from dance.transforms.graph.neighbor_graph import NeighborGraph
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell", overwrite=True)
class NeighborGraph(BaseTransform):
    """Construct neighborhood graph of observations with spatial awareness.

    This enhanced version integrates both transcriptional similarity and spatial proximity
    to create more biologically meaningful graphs for Louvain clustering.

    Parameters
    ----------
    n_neighbors
        Number of neighbors.
    n_pcs
        Number of PCs to use.
    knn
        If ``True``, then use a hard threshold to restrict the number of neighbors to ``n_neighbors``.
    random_state
        Random seed.
    method
        Method for computing the connectivities.
    metric
        Distance metric.
    channel
        Name of the PC channel.
    spatial_key
        Key for spatial coordinates in obsm.
    alpha
        Weight for combining spatial and transcriptional graphs (0 = transcriptional only, 1 = spatial only).
    use_squidpy
        Whether to use squidpy for spatial graph construction.
    spatial_radius
        Radius for spatial graph construction when using squidpy.

    """

    _DISPLAY_ATTRS = ("n_neighbors", "n_pcs", "knn", "random_state", "method", "metric", "spatial_key", "alpha",
                      "use_squidpy", "spatial_radius")

    def __init__(self, n_neighbors: int = 15, *, n_pcs: Optional[int] = None, knn: bool = True, random_state: int = 0,
                 method: Optional[str] = "umap", metric: str = "euclidean", channel: Optional[str] = "CellPCA",
                 spatial_key: Optional[str] = "spatial", alpha: float = 0.5, use_squidpy: bool = True,
                 spatial_radius: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.knn = knn
        self.random_state = random_state
        self.method = method
        self.metric = metric
        self.channel = channel
        self.spatial_key = spatial_key
        self.alpha = alpha
        self.use_squidpy = use_squidpy
        self.spatial_radius = spatial_radius

    def __call__(self, data):
        # Check if spatial data is available
        has_spatial = self.spatial_key in data.data.obsm

        if not has_spatial:
            # Fallback to original behavior when no spatial data
            self.logger.info("No spatial data found, using transcriptional graph only")
            adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel, n_neighbors=self.n_neighbors,
                                  n_pcs=self.n_pcs, knn=self.knn, random_state=self.random_state, method=self.method,
                                  metric=self.metric).obsp["connectivities"]
        else:
            # Use spatial-aware graph construction
            if self.use_squidpy:
                try:
                    import squidpy as sq

                    # Create spatial graph using squidpy with radius-based approach
                    sq.gr.spatial_neighbors(data.data, coord_type="generic", radius=self.spatial_radius,
                                            key_added="spatial_graph")

                    # Get the spatial connectivities
                    spatial_adj = data.data.obsp["spatial_graph"]

                    # Create transcriptional graph using PCA
                    trans_adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel,
                                                n_neighbors=self.n_neighbors, n_pcs=self.n_pcs, knn=self.knn,
                                                random_state=self.random_state, method=self.method,
                                                metric=self.metric).obsp["connectivities"]

                    # Combine graphs with weighted average - implement more sophisticated combination
                    if self.alpha < 1.0:
                        # Blend spatial and transcriptional graphs with distance-based weighting
                        # First normalize both matrices separately to ensure equal contribution
                        trans_norm = trans_adj.copy()
                        spatial_norm = spatial_adj.copy()

                        if trans_norm.nnz > 0:
                            trans_norm.data /= trans_norm.data.max()
                        if spatial_norm.nnz > 0:
                            spatial_norm.data /= spatial_norm.data.max()

                        # Weighted combination
                        adj = (1 - self.alpha) * trans_norm + self.alpha * spatial_norm

                        # Further refine by applying spatial distance-based weights
                        if 'spatial' in data.data.obsm.keys():
                            coords = data.data.obsm['spatial']
                            from scipy.spatial.distance import pdist, squareform
                            dist_matrix = squareform(pdist(coords))

                            # Apply a Gaussian kernel to spatial distances
                            sigma = np.percentile(dist_matrix[dist_matrix > 0], 25) if np.any(dist_matrix > 0) else 1.0
                            spatial_weights = np.exp(-dist_matrix**2 / (2 * sigma**2))

                            # Element-wise multiply to apply spatial weights
                            adj = adj.multiply(spatial_weights)
                            adj.eliminate_zeros()  # Remove zero entries after multiplication
                    else:
                        adj = spatial_norm

                    # Ensure adjacency matrix is symmetric and properly formatted
                    adj = adj.maximum(adj.T)  # Make symmetric
                    adj = adj.tocsr()

                    # Additional graph refinement: normalize again after combination
                    if adj.nnz > 0:
                        adj.data /= adj.data.max()

                except ImportError:
                    # Fallback to scanpy only if squidpy not available
                    self.logger.warning("squidpy not available, falling back to transcriptional graph")
                    adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel, n_neighbors=self.n_neighbors,
                                          n_pcs=self.n_pcs, knn=self.knn, random_state=self.random_state,
                                          method=self.method, metric=self.metric).obsp["connectivities"]
            else:
                # Use pure transcriptional graph but with improved parameters
                self.logger.info("Using transcriptional graph with improved parameters")
                adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel, n_neighbors=self.n_neighbors,
                                      n_pcs=self.n_pcs, knn=self.knn, random_state=self.random_state,
                                      method=self.method, metric=self.metric).obsp["connectivities"]

        data.data.obsp[self.out] = adj
        return data


# EVOLVE-BLOCK-END


def get_preprocessing_pipeline(dim: int = 50, n_neighbors: int = 17, log_level: LogLevel = "INFO",
                               save_info: bool = False):
    return Compose(
        FilterGenesMatch(prefixes=["ERCC", "MT-"]),
        AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
        AnnDataTransform(sc.pp.log1p),
        CellPCA(n_components=dim, save_info=save_info),
        NeighborGraph(n_neighbors=n_neighbors),
        SetConfig({
            "feature_channel": ["CellPCA", "NeighborGraph"],
            "feature_channel_type": ["obsm", "obsp"],
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
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=202, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--obs_nums", type=int, default=10000)
    args = parser.parse_args()

    scores = []
    inner_scores = []
    times = []  # 新增：用于记录每次运行的时间

    for seed in range(args.seed, args.seed + args.num_runs):
        start_time = time.time()  # 新增：记录单次循环的开始时间
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = Louvain(resolution=1)
        preprocessing_pipeline = get_preprocessing_pipeline(dim=args.n_components, n_neighbors=args.neighbors)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data, n_cells=args.obs_nums)
        preprocessing_pipeline(data)
        (x, adj), y = data.get_data(return_type="default")

        # Train and evaluate model
        model = Louvain(resolution=1)

        score = model.fit_score(adj, y.values.ravel())
        pred = model.predict()
        silhouette_score = resolve_score_func("silhouette")
        calinski_harabasz_score = resolve_score_func("calinski_harabasz")
        davies_bouldin_score = resolve_score_func("davies_bouldin")
        inner_scores.append(
            calculate_unified_scores({
                "silhouette": silhouette_score(x, pred),
                "calinski_harabasz": calinski_harabasz_score(x, pred),
                "davies_bouldin": davies_bouldin_score(x, pred)
            }))
        scores.append(score)

        end_time = time.time()  # 新增：记录单次循环的结束时间
        run_time = end_time - start_time
        times.append(run_time)  # 新增：保存耗时

        print(f"ARI: {score:.4f}, time: {run_time:.2f}s")  # 修改：打印单次运行时间和得分

    print(f"Louvain {args.sample_number}:")
    # 修改：在打印输出中加入 times 列表，以供 evaluator 捕获
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")  # 新增：打印平均运行时间
""" To reproduce louvain on other samples, please refer to command lines belows:
NOTE: you have to run multiple times to get best performance.

human dorsolateral prefrontal cortex sample 151673 (0.305):
$ python louvain.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676 (0.288):
$ python louvain.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507 (0.285):
$ python louvain.py --sample_number 151507
"""
