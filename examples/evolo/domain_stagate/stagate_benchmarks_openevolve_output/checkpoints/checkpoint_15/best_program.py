import argparse
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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
    """STAGATE spatial graph.

    Parameters
    ----------
    model_name
        Type of graph to construct. Currently support ``radius`` and ``knn``. See
        :class:`~sklearn.neighbors.NearestNeighbors` for more info.
    radius
        Radius parameter for ``radius_neighbors_graph``.
    n_neighbors
        Number of neighbors for ``kneighbors_graph``.
    use_expression
        Whether to include gene expression features in graph construction.
    spatial_weight
        Weight for spatial coordinates when combining with expression features.
    expr_channel
        Channel name for expression data.
    n_pcs
        Number of principal components to use.
    weight_edges
        Whether to use continuous edge weights instead of binary edges.
    sigma
        Sigma parameter for Gaussian edge weighting. If None, uses mean distance.
    use_pca
        Whether to use PCA for expression features (default True).

    """

    _MODELS = ("radius", "knn")
    _DISPLAY_ATTRS = ("model_name", "radius", "n_neighbors", "use_expression", "spatial_weight")

    def __init__(self, model_name: str = "radius", *, radius: float = 1, n_neighbors: int = 5,
                 channel: str = "spatial_pixel", channel_type: str = "obsm",
                 use_expression: bool = False, spatial_weight: float = 1.0,
                 expr_channel: str = "X_pca", n_pcs: int = 10, weight_edges: bool = False,
                 sigma: Optional[float] = None, use_pca: bool = True, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(model_name, str) or (model_name.lower() not in self._MODELS):
            raise ValueError(f"Unknown model {model_name!r}, available options are {self._MODELS}")
        self.model_name = model_name
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.channel = channel
        self.channel_type = channel_type
        self.use_expression = use_expression
        self.spatial_weight = spatial_weight
        self.expr_channel = expr_channel
        self.n_pcs = n_pcs
        self.weight_edges = weight_edges
        self.sigma = sigma
        self.use_pca = use_pca

    def __call__(self, data):
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        
        # Get expression data if needed
        if self.use_expression:
            expr_data = data.get_feature(return_type="numpy", channel=self.expr_channel)
            
            # Standardize spatial coordinates and expression data together for better scaling
            scaler = StandardScaler()
            if self.use_pca and self.expr_channel == "X_pca":
                expr_scaled = scaler.fit_transform(expr_data[:, :self.n_pcs])
            else:
                # For non-PCA expression data, take first n_pcs or all if less
                n_expr_features = min(self.n_pcs, expr_data.shape[1])
                expr_scaled = scaler.fit_transform(expr_data[:, :n_expr_features])
            
            # Scale spatial coordinates separately and apply weight
            spatial_scaler = StandardScaler()
            xy_scaled = spatial_scaler.fit_transform(xy_pixel)
            
            # Combine spatial and expression features
            combined_features = np.column_stack([
                self.spatial_weight * xy_scaled,
                expr_scaled
            ])
        else:
            combined_features = xy_pixel
            
        if self.model_name.lower() == "radius":
            nn = NearestNeighbors(radius=self.radius)
            nn.fit(combined_features)
            adj = nn.radius_neighbors_graph(combined_features)
            if self.weight_edges:
                adj = self._apply_edge_weights(adj, nn, combined_features)
        elif self.model_name.lower() == "knn":
            nn = NearestNeighbors(n_neighbors=self.n_neighbors)
            nn.fit(combined_features)
            adj = nn.kneighbors_graph(combined_features)
            if self.weight_edges:
                adj = self._apply_edge_weights(adj, nn, combined_features)

        # Ensure we're storing a scipy sparse matrix
        if not isinstance(adj, scipy.sparse.spmatrix):
            adj = scipy.sparse.csr_matrix(adj)
        data.data.obsp[self.out] = adj

    def _apply_edge_weights(self, adj, nn, combined_features):
        """Apply Gaussian edge weights to the adjacency matrix."""
        # Convert to dense for easier manipulation
        if not scipy.sparse.issparse(adj):
            adj_dense = adj
        else:
            adj_dense = adj.toarray()
        
        # Get distances for all connections
        distances, indices = nn.kneighbors(combined_features, return_distance=True)
        
        # Compute sigma if not provided
        if self.sigma is None:
            # Use median of all distances for stability
            all_distances = []
            for i in range(len(distances)):
                all_distances.extend(distances[i][1:])  # Skip self-connections
            sigma = np.median(all_distances) if len(all_distances) > 0 else 1.0
        else:
            sigma = self.sigma
        
        # Vectorized application of Gaussian weights
        for i in range(len(distances)):
            # Get indices and distances for node i (excluding self)
            node_indices = indices[i][1:]
            node_distances = distances[i][1:]
            
            # Only update if there are neighbors
            if len(node_indices) > 0:
                # Apply Gaussian weights
                weights = np.exp(-node_distances**2 / (2 * sigma**2))
                
                # Update the adjacency matrix
                adj_dense[i, node_indices] = weights
        
        # Return as sparse matrix
        return scipy.sparse.csr_matrix(adj_dense)
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
