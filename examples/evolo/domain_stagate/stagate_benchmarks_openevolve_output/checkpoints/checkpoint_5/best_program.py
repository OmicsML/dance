import argparse
import time  # 新增：导入 time 模块

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.sparse

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
        Number of principal components to use for expression features.
    weight_edges
        Whether to use continuous edge weights instead of binary weights.
    sigma
        Sigma parameter for Gaussian edge weighting.

    """

    _MODELS = ("radius", "knn", "delaunay", "gabriel")
    _DISPLAY_ATTRS = ("model_name", "radius", "n_neighbors", "use_expression", "spatial_weight")

    def __init__(self, model_name: str = "radius", *, radius: float = 1, n_neighbors: int = 5,
                 channel: str = "spatial_pixel", channel_type: str = "obsm",
                 use_expression: bool = False, spatial_weight: float = 1.0,
                 expr_channel: str = "X_pca", n_pcs: int = 10,
                 weight_edges: bool = False, sigma: float = None, **kwargs):
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

    def __call__(self, data):
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        
        # Handle expression features if requested
        if self.use_expression:
            # Get expression data
            expr_data = data.get_feature(return_type="numpy", channel=self.expr_channel, channel_type="obsm")
            
            # Apply PCA if needed
            if expr_data.shape[1] > self.n_pcs:
                pca = PCA(n_components=self.n_pcs)
                expr_pcs = pca.fit_transform(expr_data)
            else:
                expr_pcs = expr_data
            
            # Scale spatial coordinates and expression PCs
            scaler_spatial = StandardScaler()
            scaler_expr = StandardScaler()
            
            xy_scaled = scaler_spatial.fit_transform(xy_pixel)
            expr_scaled = scaler_expr.fit_transform(expr_pcs)
            
            # Combine spatial and expression features
            combined_features = np.hstack([
                self.spatial_weight * xy_scaled,
                expr_scaled
            ])
        else:
            # Use only spatial coordinates
            combined_features = xy_pixel

        # Build graph based on specified model
        if self.model_name.lower() == "radius":
            nn = NearestNeighbors(radius=self.radius)
            nn.fit(combined_features)
            adj = nn.radius_neighbors_graph(combined_features)
        elif self.model_name.lower() == "knn":
            nn = NearestNeighbors(n_neighbors=self.n_neighbors)
            nn.fit(combined_features)
            adj = nn.kneighbors_graph(combined_features)
        elif self.model_name.lower() == "delaunay":
            try:
                from scipy.spatial import Delaunay
                # Use only spatial coordinates for triangulation
                if self.use_expression:
                    spatial_only = xy_pixel
                else:
                    spatial_only = combined_features[:, :2] if combined_features.shape[1] >= 2 else combined_features
                
                # Perform Delaunay triangulation
                tri = Delaunay(spatial_only)
                
                # Create adjacency matrix from triangulation
                rows, cols = [], []
                for simplex in tri.simplices:
                    # Add edges between all pairs in the simplex
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            rows.extend([simplex[i], simplex[j]])
                            cols.extend([simplex[j], simplex[i]])
                
                # Create symmetric adjacency matrix
                adj = scipy.sparse.coo_matrix(
                    (np.ones(len(rows)), (rows, cols)),
                    shape=(len(combined_features), len(combined_features))
                ).tocsr()
            except Exception:
                # Fallback to KNN if Delaunay fails (e.g., due to collinear points)
                print("Delaunay triangulation failed, falling back to KNN")
                nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(combined_features)-1))
                nn.fit(combined_features)
                adj = nn.kneighbors_graph(combined_features)
        elif self.model_name.lower() == "gabriel":
            try:
                from scipy.spatial.distance import pdist, squareform
                from scipy.spatial import Delaunay
                # Use only spatial coordinates for Gabriel graph construction
                if self.use_expression:
                    spatial_only = xy_pixel
                else:
                    spatial_only = combined_features[:, :2] if combined_features.shape[1] >= 2 else combined_features
                
                # First compute Delaunay triangulation
                tri = Delaunay(spatial_only)
                
                # Create Gabriel graph from Delaunay - keep edge only if no other point in circle
                rows, cols = [], []
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            p1_idx, p2_idx = simplex[i], simplex[j]
                            p1, p2 = spatial_only[p1_idx], spatial_only[p2_idx]
                            
                            # Midpoint and radius of circle
                            midpoint = (p1 + p2) / 2
                            radius = np.linalg.norm(p1 - p2) / 2
                            
                            # Check if any other point is inside the circle
                            valid_edge = True
                            center_to_point_sq = np.sum((spatial_only - midpoint)**2, axis=1)
                            radius_sq = radius**2
                            
                            # Points within the circle (excluding p1 and p2)
                            mask = (center_to_point_sq < radius_sq) & (np.arange(len(spatial_only)) != p1_idx) & (np.arange(len(spatial_only)) != p2_idx)
                            
                            if not np.any(mask):
                                # Valid Gabriel edge
                                rows.extend([p1_idx, p2_idx])
                                cols.extend([p2_idx, p1_idx])
                
                # Create symmetric adjacency matrix
                adj = scipy.sparse.coo_matrix(
                    (np.ones(len(rows)), (rows, cols)),
                    shape=(len(combined_features), len(combined_features))
                ).tocsr()
            except Exception:
                # Fallback to KNN if Gabriel graph fails
                print("Gabriel graph construction failed, falling back to KNN")
                nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(combined_features)-1))
                nn.fit(combined_features)
                adj = nn.kneighbors_graph(combined_features)

        # Apply continuous edge weighting if requested
        if self.weight_edges:
            # Convert to dense array to compute distances
            adj_dense = adj.toarray()
            
            # Compute full distance matrix for connected nodes based on model type
            if self.model_name.lower() in ["knn", "radius"]:
                if self.model_name.lower() == "knn":
                    distances_full = nn.kneighbors(combined_features, return_distance=True)[0]
                elif self.model_name.lower() == "radius":
                    distances_full = nn.radius_neighbors(combined_features, return_distance=True)[0]
                    
                # Compute Gaussian weights
                if self.sigma is None:
                    # Use mean distance of neighbors as sigma
                    all_distances = []
                    for i in range(len(combined_features)):
                        neighbors = adj[i].nonzero()[1]
                        if len(neighbors) > 0:
                            dists = np.linalg.norm(
                                combined_features[i:i+1] - combined_features[neighbors],
                                axis=1
                            )
                            all_distances.extend(dists)
                    
                    if len(all_distances) > 0:
                        self.sigma = np.mean(all_distances)
                    else:
                        self.sigma = 1.0
                
                # Create weighted adjacency matrix
                weighted_adj = np.zeros_like(adj_dense, dtype=np.float32)
                
                # For each node, compute weights based on distances to neighbors
                for i in range(len(combined_features)):
                    neighbors = adj[i].nonzero()[1]
                    if len(neighbors) > 0:
                        # Get distances for this node's neighbors
                        dists = np.linalg.norm(
                            combined_features[i:i+1] - combined_features[neighbors], 
                            axis=1
                        )
                        # Apply Gaussian kernel
                        weights = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
                        weighted_adj[i, neighbors] = weights
            else:
                # For Delaunay and Gabriel graphs, compute distances manually
                if self.sigma is None:
                    # Estimate sigma from average distance of edges
                    all_distances = []
                    rows, cols = adj.nonzero()
                    for r, c in zip(rows, cols):
                        if r < c:  # Avoid double counting
                            dist = np.linalg.norm(combined_features[r] - combined_features[c])
                            all_distances.append(dist)
                    
                    if len(all_distances) > 0:
                        self.sigma = np.mean(all_distances)
                    else:
                        self.sigma = 1.0
                
                # Create weighted adjacency matrix
                weighted_adj = np.zeros_like(adj_dense, dtype=np.float32)
                
                # For each node, compute weights based on distances to neighbors
                for i in range(len(combined_features)):
                    neighbors = adj[i].nonzero()[1]
                    if len(neighbors) > 0:
                        # Get distances for this node's neighbors
                        dists = np.linalg.norm(
                            combined_features[i:i+1] - combined_features[neighbors], 
                            axis=1
                        )
                        # Apply Gaussian kernel
                        weights = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
                        weighted_adj[i, neighbors] = weights
            
            # Convert back to sparse format
            adj = scipy.sparse.csr_matrix(weighted_adj)
        
        data.data.obsp[self.out] = adj
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
    times = []  # 新增：用于记录每次运行的时间

    for seed in range(args.seed, args.seed + args.num_runs):
        start_time = time.time()  # 新增：记录单次循环的开始时间
        
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
        
        end_time = time.time()  # 新增：记录单次循环的结束时间
        run_time = end_time - start_time
        times.append(run_time)  # 新增：保存耗时
        
        print(f"ARI: {score:.4f}, time: {run_time:.2f}s")  # 修改：同时输出得分与时间
        
    print(f"STAGATE {args.sample_number}:")
    # 修改：加入 times 列表，以供 evaluator 捕获
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")  # 新增：输出平均运行时间

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