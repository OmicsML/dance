import argparse
import scipy.sparse

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.graph.spatial_graph import StagateGraph
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils.metrics import calculate_unified_scores, resolve_score_func
import numpy as np
import scanpy as sc
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stagate import Stagate
from dance.utils import set_seed, sub_data
from dance.typing import Optional


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "spatial",overwrite=True)
class StagateGraph(BaseTransform):
    """STAGATE spatial graph with enhanced hybrid spatial-expression capabilities.

    Parameters
    ----------
    model_name
        Type of graph to construct. Currently support ``radius`` and ``knn``. See
        :class:`~sklearn.neighbors.NearestNeighbors` for more info.
    radius
        Radius parameter for ``radius_neighbors_graph``.
    n_neighbors
        Number of neighbors for ``kneighbors_graph``.
    spatial_weight
        Weight factor for spatial coordinates when combining with expression features.
    use_expression
        Whether to incorporate expression features into graph construction.
    expr_channel
        Channel name for expression features (e.g., 'CellPCA').
    expr_channel_type
        Channel type for expression features ('obsm', 'X', etc.).
    n_pcs
        Number of principal components to use when expression features are used.
    weight_edges
        Whether to use continuous edge weights instead of binary edges.
    sigma
        Sigma parameter for Gaussian edge weighting. If None, uses mean distance.
    use_pca
        Whether to perform PCA on expression features (default True).

    """

    _MODELS = ("radius", "knn")
    _DISPLAY_ATTRS = ("model_name", "radius", "n_neighbors", "spatial_weight", "use_expression")

    def __init__(self, model_name: str = "radius", *, radius: float = 1, n_neighbors: int = 5,
                 channel: str = "spatial_pixel", channel_type: str = "obsm", 
                 spatial_weight: float = 1.0, use_expression: bool = False, 
                 expr_channel: str = "CellPCA", expr_channel_type: str = "obsm",
                 n_pcs: int = 10, weight_edges: bool = False, sigma: Optional[float] = None,
                 use_pca: bool = True, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(model_name, str) or (model_name.lower() not in self._MODELS):
            raise ValueError(f"Unknown model {model_name!r}, available options are {self._MODELS}")
        self.model_name = model_name
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.channel = channel
        self.channel_type = channel_type
        self.spatial_weight = spatial_weight
        self.use_expression = use_expression
        self.expr_channel = expr_channel
        self.expr_channel_type = expr_channel_type
        self.n_pcs = n_pcs
        self.weight_edges = weight_edges
        self.sigma = sigma
        self.use_pca = use_pca

    def __call__(self, data):
        # Get spatial coordinates
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        
        # If using expression features, combine with spatial coordinates
        if self.use_expression:
            expr_features = data.get_feature(return_type="numpy", channel=self.expr_channel, channel_type=self.expr_channel_type)
            
            # Standardize spatial coordinates
            scaler_spatial = StandardScaler()
            xy_scaled = scaler_spatial.fit_transform(xy_pixel)
            
            # Handle expression features with PCA if requested
            if self.use_pca and self.expr_channel == "CellPCA":
                # Already in PCA space, just use specified number of components
                expr_scaled = expr_features[:, :self.n_pcs]
            else:
                # Perform PCA on expression data if needed
                if self.use_pca:
                    pca = PCA(n_components=min(self.n_pcs, expr_features.shape[1]))
                    expr_pca = pca.fit_transform(expr_features)
                    expr_scaled = expr_pca[:, :self.n_pcs]
                else:
                    # Just use first n_pcs of expression features
                    expr_scaled = expr_features[:, :self.n_pcs]
            
            # Standardize expression features
            scaler_expr = StandardScaler()
            expr_scaled = scaler_expr.fit_transform(expr_scaled)
            
            # Apply spatial weight and combine with expression features
            xy_weighted = xy_scaled * self.spatial_weight
            combined_features = np.hstack([xy_weighted, expr_scaled])
            
            features_to_use = combined_features
        else:
            features_to_use = xy_pixel

        if self.model_name.lower() == "radius":
            nn = NearestNeighbors(radius=self.radius)
            nn.fit(features_to_use)
            adj = nn.radius_neighbors_graph(features_to_use)
            if self.weight_edges:
                adj = self._apply_edge_weights(adj, nn, features_to_use)
        elif self.model_name.lower() == "knn":
            nn = NearestNeighbors(n_neighbors=self.n_neighbors)
            nn.fit(features_to_use)
            adj = nn.kneighbors_graph(features_to_use)
            if self.weight_edges:
                adj = self._apply_edge_weights(adj, nn, features_to_use)

        # Ensure we're storing a scipy sparse matrix
        if not isinstance(adj, scipy.sparse.spmatrix):
            adj = scipy.sparse.csr_matrix(adj)
        data.data.obsp[self.out] = adj

    def _apply_edge_weights(self, adj, nn, features_to_use):
        """Apply Gaussian edge weights to the adjacency matrix."""
        # Get distances for all connections efficiently
        distances, indices = nn.kneighbors(features_to_use, return_distance=True)
        
        # Compute sigma if not provided
        if self.sigma is None:
            # Use median of all distances for stability
            all_distances = []
            for i in range(len(distances)):
                all_distances.extend(distances[i][1:])  # Skip self-connections
            sigma = np.median(all_distances) if len(all_distances) > 0 else 1.0
        else:
            sigma = self.sigma
        
        # Vectorized application of Gaussian weights using sparse matrix construction
        rows, cols, weights = [], [], []
        
        for i in range(len(distances)):
            # Get indices and distances for node i (excluding self)
            node_indices = indices[i][1:]
            node_distances = distances[i][1:]
            
            # Only process if there are neighbors
            if len(node_indices) > 0:
                # Apply Gaussian weights
                node_weights = np.exp(-node_distances**2 / (2 * sigma**2))
                
                # Add to sparse matrix data
                rows.extend([i] * len(node_indices))
                cols.extend(node_indices)
                weights.extend(node_weights)
        
        # Create sparse matrix directly (more efficient than dense conversion)
        return scipy.sparse.csr_matrix((weights, (rows, cols)), 
                                      shape=adj.shape)
# EVOLVE-BLOCK-END

def get_preprocessing_pipeline(hvg_flavor: str = "seurat_v3", n_top_hvgs: int = 3000, model_name: str = "radius",
                               radius: float = 150, n_neighbors: int = 5, log_level: LogLevel = "INFO"):
        return Compose(
            AnnDataTransform(sc.pp.highly_variable_genes, flavor=hvg_flavor, n_top_genes=n_top_hvgs, subset=True),
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            StagateGraph(model_name, radius=radius, n_neighbors=n_neighbors),
            SetConfig({
                "feature_channel": "StagateGraph",
                "feature_channel_type": "obsp",
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
    parser.add_argument("--hidden_dims", type=list, default=[512, 32], help="hidden dimensions")
    parser.add_argument("--rad_cutoff", type=int, default=150, help="")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--high_variable_genes", type=int, default=3000, help="")
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0').")
    parser.add_argument("--obs_nums",type=int,default=10000)
    args = parser.parse_args()

    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        preprocessing_pipeline = get_preprocessing_pipeline(n_top_hvgs=args.high_variable_genes,
                                                              radius=args.rad_cutoff)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data,args.obs_nums)
        preprocessing_pipeline(data)
        adj, y = data.get_data(return_type="default")
        x = data.data.X.A
        edge_list_array = np.vstack(np.nonzero(adj))

        # Train and evaluate model
        model = Stagate([min(args.high_variable_genes,x.shape[1])] + args.hidden_dims,device=args.device)
        score = model.fit_score((x, edge_list_array), y, epochs=args.epochs, random_state=seed)
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
    print(f"STAGATE {args.sample_number}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
""" To reproduce Stagate on other samples, please refer to command lines belows:
NOTE: since the stagate method is unstable, you have to run at least 5 times to get
      best performance. (same with original Stagate paper)

human dorsolateral prefrontal cortex sample 151673:
$ python stagate.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python stagate.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python stagate.py --sample_number 151507
"""
