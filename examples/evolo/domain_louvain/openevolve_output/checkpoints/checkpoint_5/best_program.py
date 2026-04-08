import argparse
from typing import Optional

import numpy as np
import scanpy as sc

from dance.transforms.filter import FilterGenesMatch

try:
    import squidpy as sq
    HAS_SQUIDPY = True
except ImportError:
    HAS_SQUIDPY = False

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
    """Construct neighborhood graph of observations with spatial awareness.

    This enhanced version incorporates both transcriptional similarity and spatial proximity
    to create a more biologically meaningful graph for Louvain clustering.

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
        Weight for spatial component when combining graphs (0.0 = transcriptional only, 1.0 = spatial only).
    use_spatial
        Whether to use spatial information (requires squidpy).

    """

    _DISPLAY_ATTRS = ("n_neighbors", "n_pcs", "knn", "random_state", "method", "metric", "spatial_weight",
                      "use_spatial")

    def __init__(self, n_neighbors: int = 15, *, n_pcs: Optional[int] = None, knn: bool = True, random_state: int = 0,
                 method: Optional[str] = "umap", metric: str = "euclidean", channel: Optional[str] = "CellPCA",
                 spatial_weight: float = 0.5, use_spatial: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.knn = knn
        self.random_state = random_state
        self.method = method
        self.metric = metric
        self.channel = channel
        self.spatial_weight = spatial_weight
        self.use_spatial = use_spatial

    def __call__(self, data):
        self.logger.info("Start computing the kNN connectivity adjacency matrix with spatial awareness")

        # Get transcriptional graph (original approach)
        if self.channel in data.data.obsm:
            trans_adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel, n_neighbors=self.n_neighbors,
                                        n_pcs=self.n_pcs, knn=self.knn, random_state=self.random_state,
                                        method=self.method, metric=self.metric).obsp["connectivities"]
        else:
            # Fallback if channel doesn't exist
            trans_adj = sc.pp.neighbors(data.data, copy=True, n_neighbors=self.n_neighbors, knn=self.knn,
                                        random_state=self.random_state, method=self.method,
                                        metric=self.metric).obsp["connectivities"]

        # If we have spatial data and squidpy is available, create spatial graph
        if (self.use_spatial and HAS_SQUIDPY and 'spatial' in data.data.obsm
                and hasattr(data.data.obsm['spatial'], '__array__')):

            # Calculate an adaptive radius based on the median distance between cells
            spatial_coords = data.data.obsm['spatial']
            if len(spatial_coords) > 1:
                # Compute pairwise distances and use median as adaptive radius
                from scipy.spatial.distance import pdist
                dists = pdist(spatial_coords)
                adaptive_radius = min(np.percentile(dists, 25) * 3, 100.0)  # Cap at 100 to avoid too dense graphs
            else:
                adaptive_radius = 50.0  # Fallback

            # Create spatial graph using squidpy with adaptive radius
            try:
                # Use squidpy for spatial neighbor graph construction with distance-weighted edges
                sq.gr.spatial_neighbors(data.data, coord_type='generic', radius=adaptive_radius,
                                        n_neighs=self.n_neighbors)
                spatial_adj = data.data.obsp['spatial_connectivities']

                # For better combination, first normalize each adjacency matrix independently
                if self.spatial_weight > 0:
                    # Convert to dense for normalization and combination
                    trans_dense = trans_adj.toarray()
                    spatial_dense = spatial_adj.toarray()

                    # Normalize each adjacency matrix to have similar magnitude ranges
                    if trans_dense.max() > 0:
                        trans_dense = trans_dense / trans_dense.max()
                    if spatial_dense.max() > 0:
                        # Apply distance-based weighting to spatial adjacency (closer cells get higher weights)
                        from scipy.spatial.distance import cdist
                        coords = data.data.obsm['spatial']
                        spatial_dist = cdist(coords, coords)
                        # Create distance mask - higher values for closer neighbors
                        max_dist_in_spatial = spatial_dist[spatial_dense > 0].max() if spatial_dist[spatial_dense >
                                                                                                    0].size > 0 else 1.0
                        distance_weight = np.where(spatial_dist > 0, np.exp(-spatial_dist / max_dist_in_spatial), 0)

                        # Apply the distance weight to spatial adjacency
                        spatial_dense = spatial_dense * distance_weight
                        # Renormalize after distance weighting
                        if spatial_dense.max() > 0:
                            spatial_dense = spatial_dense / spatial_dense.max()

                    # Weighted combination
                    combined_dense = (1 - self.spatial_weight) * trans_dense + self.spatial_weight * spatial_dense

                    # Convert back to sparse format
                    from scipy.sparse import csr_matrix
                    combined_adj = csr_matrix(combined_dense)

                    # Remove very weak connections to reduce noise (graph pruning)
                    # Only keep edges that are above a certain threshold of the maximum connection
                    threshold = 0.05 * combined_adj.max()  # Keep edges with at least 5% of max weight
                    combined_adj.data[combined_adj.data < threshold] = 0
                    combined_adj.eliminate_zeros()

                    # Ensure symmetry and remove self-loops
                    combined_adj = combined_adj.maximum(combined_adj.T)
                    combined_adj.setdiag(0)
                    combined_adj.eliminate_zeros()

                else:
                    # Convert to sparse for consistency
                    from scipy.sparse import csr_matrix
                    combined_adj = csr_matrix(trans_adj.toarray())

            except Exception as e:
                self.logger.warning(f"Squidpy spatial graph construction failed: {e}, using transcriptional graph only")
                from scipy.sparse import csr_matrix
                combined_adj = csr_matrix(trans_adj.toarray())

        else:
            # Use transcriptional graph only
            from scipy.sparse import csr_matrix
            combined_adj = csr_matrix(trans_adj.toarray())

        data.data.obsp[self.out] = combined_adj

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
    args = parser.parse_args()

    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = Louvain(resolution=1)
        preprocessing_pipeline = get_preprocessing_pipeline(dim=args.n_components, n_neighbors=args.neighbors)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data)
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
        print(f"ARI: {score:.4f}")
    print(f"Louvain {args.sample_number}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
""" To reproduce louvain on other samples, please refer to command lines belows:
NOTE: you have to run multiple times to get best performance.

human dorsolateral prefrontal cortex sample 151673 (0.305):
$ python louvain.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676 (0.288):
$ python louvain.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507 (0.285):
$ python louvain.py --sample_number 151507
"""
