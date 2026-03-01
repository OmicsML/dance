import argparse
from typing import Optional

from dance.transforms.filter import FilterGenesMatch
import numpy as np
import scanpy as sc
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
from dance.utils import set_seed,sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func

# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell",overwrite=True)
class NeighborGraph(BaseTransform):
    """Construct neighborhood graph of observations with spatial awareness.

    This implementation enhances the standard kNN graph construction by incorporating
    spatial information from tissue coordinates, creating a more biologically meaningful
    graph for Louvain community detection.

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
    use_spatial
        Whether to use spatial information when constructing the graph.
    alpha
        Weight for combining spatial and transcriptional graphs (0 = transcriptional only, 1 = spatial only).
    spatial_key
        Key for spatial coordinates in obsm.

    """

    _DISPLAY_ATTRS = ("n_neighbors", "n_pcs", "knn", "random_state", "method", "metric", "use_spatial", "alpha")

    def __init__(self, n_neighbors: int = 15, *, n_pcs: Optional[int] = None, knn: bool = True, random_state: int = 0,
                 method: Optional[str] = "umap", metric: str = "euclidean", channel: Optional[str] = "CellPCA",
                 use_spatial: bool = True, alpha: float = 0.5, spatial_key: str = "spatial", **kwargs):
        super().__init__(**kwargs)

        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.knn = knn
        self.random_state = random_state
        self.method = method
        self.metric = metric
        self.channel = channel
        self.use_spatial = use_spatial
        self.alpha = alpha
        self.spatial_key = spatial_key

    def __call__(self, data):
        self.logger.info("Start computing the kNN connectivity adjacency matrix with spatial enhancement")
        
        # Get the base transcriptional graph
        trans_adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel, n_neighbors=self.n_neighbors,
                                    n_pcs=self.n_pcs, knn=self.knn, random_state=self.random_state, method=self.method,
                                    metric=self.metric).obsp["connectivities"]
        
        if not self.use_spatial or not HAS_SQUIDPY:
            # Fall back to standard approach if spatial not available or squidpy not installed
            data.data.obsp[self.out] = trans_adj
            return data
            
        # When spatial information is available, create spatial graph
        if self.spatial_key in data.data.obsm:
            try:
                # Create spatial graph using squidpy
                sq.gr.spatial_neighbors(data.data, coord_type="generic", spatial_key=self.spatial_key, 
                                      n_neighs=self.n_neighbors * 2, key_added="spatial_graph")
                
                # Get spatial adjacency matrix
                spatial_adj = data.data.obsp["spatial_graph_connectivities"]
                
                # Combine graphs: weighted average
                if self.alpha > 0 and self.alpha < 1:
                    # Blend spatial and transcriptional graphs
                    combined_adj = self.alpha * spatial_adj + (1 - self.alpha) * trans_adj
                elif self.alpha == 1:
                    # Use spatial graph only
                    combined_adj = spatial_adj
                else:
                    # Use transcriptional graph only
                    combined_adj = trans_adj
                
                # Ensure the result is properly formatted as a sparse matrix
                from scipy.sparse import csr_matrix
                if hasattr(combined_adj, 'toarray'):
                    combined_adj = csr_matrix(combined_adj)
                
                data.data.obsp[self.out] = combined_adj
                
            except Exception as e:
                # Fallback to transcriptional graph if spatial processing fails
                self.logger.warning(f"Spatial graph construction failed: {e}, falling back to transcriptional graph")
                data.data.obsp[self.out] = trans_adj
        else:
            # No spatial coordinates available, use transcriptional graph only
            data.data.obsp[self.out] = trans_adj

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
    parser.add_argument("--obs_nums",type=int,default=10000)
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
        sub_data(data.data,n_cells=args.obs_nums)
        preprocessing_pipeline(data)
        (x,adj), y = data.get_data(return_type="default")

        # Train and evaluate model
        model = Louvain(resolution=1)
        
        score = model.fit_score(adj, y.values.ravel())
        pred=model.predict()
        silhouette_score = resolve_score_func("silhouette")
        calinski_harabasz_score = resolve_score_func("calinski_harabasz")
        davies_bouldin_score = resolve_score_func("davies_bouldin")
        inner_scores.append(calculate_unified_scores({
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
