import argparse
import time  # 新增：导入 time 模块
from typing import Optional

import numpy as np
import scanpy as sc

from dance.transforms.filter import FilterGenesMatch

try:
    import squidpy as sq
except ImportError:
    sq = None

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.louvain import Louvain
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA
from dance.transforms.graph.neighbor_graph import NeighborGraph
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell", overwrite=True)
class NeighborGraph(BaseTransform):
    """Construct neighborhood graph of observations.

    This is a thin wrapper that creates a combined graph based on both transcriptional similarity
    and spatial proximity using scanpy and optionally squidpy for spatial information.

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
    spatial_weight
        Weight for spatial graph when combining with transcriptional graph (0 to 1).
    radius
        Radius for spatial neighbor search if using radius-based graph construction.

    """

    _DISPLAY_ATTRS = ("n_neighbors", "n_pcs", "knn", "random_state", "method", "metric", "spatial_weight", "radius")

    def __init__(self, n_neighbors: int = 15, *, n_pcs: Optional[int] = None, knn: bool = True, random_state: int = 0,
                 method: Optional[str] = "umap", metric: str = "euclidean", channel: Optional[str] = "CellPCA",
                 spatial_weight: float = 0.5, radius: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)

        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.knn = knn
        self.random_state = random_state
        self.method = method
        self.metric = metric
        self.channel = channel
        self.spatial_weight = spatial_weight
        self.radius = radius

    def __call__(self, data):
        self.logger.info("Start computing the combined transcriptional-spatial connectivity adjacency matrix")

        # Compute transcriptional graph using PCA features
        trans_adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel, n_neighbors=self.n_neighbors,
                                    n_pcs=self.n_pcs, knn=self.knn, random_state=self.random_state, method=self.method,
                                    metric=self.metric).obsp["connectivities"]

        # Normalize transcriptional adjacency matrix
        trans_adj = trans_adj / np.max(trans_adj.data) if trans_adj.nnz > 0 else trans_adj

        # Compute spatial graph if possible
        spatial_adj = None
        if sq is not None and 'spatial' in data.data.obsm.keys():
            try:
                # Use squidpy for spatial graph construction
                if self.radius is not None:
                    # Create radius-based spatial graph
                    sq.gr.spatial_neighbors(data.data, coord_type="generic", radius=self.radius)
                    spatial_adj = data.data.obsp['spatial_connectivities']
                else:
                    # Create k-nearest neighbors spatial graph
                    sq.gr.spatial_neighbors(data.data, coord_type="generic", n_neigh=self.n_neighbors)
                    spatial_adj = data.data.obsp['spatial_connectivities']

                # Normalize spatial adjacency matrix
                spatial_adj = spatial_adj / np.max(spatial_adj.data) if spatial_adj.nnz > 0 else spatial_adj
            except Exception as e:
                self.logger.warning(f"Spatial graph computation failed: {e}. Using only transcriptional graph.")
                spatial_adj = None

        # Combine the graphs based on spatial weight
        if spatial_adj is not None:
            # Ensure both matrices have the same shape
            if trans_adj.shape != spatial_adj.shape:
                self.logger.warning(
                    "Transcriptional and spatial adjacency matrices have different shapes. Using only transcriptional graph."
                )
                final_adj = trans_adj
            else:
                # Combine with specified weights
                final_adj = (1 - self.spatial_weight) * trans_adj + self.spatial_weight * spatial_adj
        else:
            final_adj = trans_adj

        # Convert to symmetric matrix to ensure proper behavior for Louvain clustering
        final_adj = final_adj.maximum(final_adj.T)

        # Store the final combined adjacency matrix
        data.data.obsp[self.out] = final_adj

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
