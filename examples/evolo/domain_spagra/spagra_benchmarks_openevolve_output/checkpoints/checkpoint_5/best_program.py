import argparse
import time  # 新增：导入 time 模块

import numpy as np
import pandas as pd
import sklearn

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spaGRA import SpaGRA
from dance.registry import register_preprocessor
from dance.transforms import Compose, HighlyVariableGenesRawCount, NormalizeTotalLog1P, PrefilterGenes, SetConfig
from dance.transforms.base import BaseTransform
from dance.typing import LogLevel, Optional
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "spatial", overwrite=True)
class CalSpatialNet(BaseTransform):
    """Construct the spatial neighbor networks.

    Parameters
    ----------
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. Options include 'Radius', 'KNN', 'Delaunay', 'Gaussian'.
        When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff.
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
        When model=='Delaunay', uses Delaunay triangulation for neighbor connections.
        When model=='Gaussian', uses Gaussian kernel weights based on distances.
    spatial_uns
        Key for storing spatial networks in adata.uns. Default is "Spatial_Net".

    """

    _DISPLAY_ATTRS = ("rad_cutoff", "k_cutoff", "model", "spatial_uns")

    def __init__(self, rad_cutoff=None, k_cutoff=None, model='Radius', spatial_uns="Spatial_Net", out=None,
                 log_level="WARNING", sigma=None, knn_weighted=False):
        super().__init__(out=out, log_level=log_level)
        self.rad_cutoff = rad_cutoff
        self.k_cutoff = k_cutoff
        self.model = model
        self.spatial_uns = spatial_uns
        self.sigma = sigma
        self.knn_weighted = knn_weighted  # Option to add Gaussian weights to KNN graph

    def __call__(self, data):
        """Construct spatial neighbor networks."""
        adata = data.data

        print('------Calculating spatial graph...')
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']

        n_cells = coor.shape[0]

        if self.model == 'Radius':
            # Vectorized radius neighbor search
            nbrs = sklearn.neighbors.NearestNeighbors(radius=self.rad_cutoff).fit(coor)
            distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

            # Vectorized edge creation
            cell1_ids = np.concatenate([np.full(len(indices[i]), i) for i in range(n_cells)])
            cell2_ids = np.concatenate(indices)
            distances = np.concatenate(distances)

            # Remove self-loops and zero distances
            mask = (cell1_ids != cell2_ids) & (distances > 0)
            cell1_ids = cell1_ids[mask]
            cell2_ids = cell2_ids[mask]
            distances = distances[mask]

        elif self.model == 'KNN':
            # Vectorized KNN search
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k_cutoff + 1).fit(coor)
            distances, indices = nbrs.kneighbors(coor)

            # Remove self-loops (first neighbor is always self)
            indices = indices[:, 1:]
            distances = distances[:, 1:]

            # Flatten arrays for vectorized operation
            cell1_ids = np.repeat(np.arange(n_cells), self.k_cutoff)
            cell2_ids = indices.flatten()
            distances = distances.flatten()

        elif self.model == 'Delaunay':
            # Use Delaunay triangulation for better tissue topology representation
            from scipy.spatial import Delaunay
            from scipy.spatial.distance import pdist, squareform

            # Handle edge cases for small datasets
            if n_cells < 4:
                # Fall back to KNN for very small datasets
                return self._create_knn_graph(coor, n_cells)

            try:
                delaunay = Delaunay(coor.values)
                simplices = delaunay.simplices

                # Get all edges from simplices (triangles)
                edges = set()
                for simplex in simplices:
                    # Create edges from triangle vertices
                    for i in range(3):
                        for j in range(i + 1, 3):
                            edge = tuple(sorted([simplex[i], simplex[j]]))
                            edges.add(edge)

                # Convert to arrays
                edge_list = list(edges)
                if len(edge_list) > 0:
                    cell1_ids = np.array([edge[0] for edge in edge_list])
                    cell2_ids = np.array([edge[1] for edge in edge_list])

                    # Calculate actual distances for consistency with interface
                    distances = np.sqrt(np.sum((coor.iloc[cell1_ids].values - coor.iloc[cell2_ids].values)**2, axis=1))
                else:
                    # If Delaunay failed to produce edges, fall back to KNN
                    return self._create_knn_graph(coor, n_cells)

            except Exception:
                # Fallback to KNN if Delaunay fails
                return self._create_knn_graph(coor, n_cells)

        elif self.model == 'Gaussian':
            # Use Gaussian kernel weights
            if self.sigma is None:
                # Adaptive sigma based on median distance
                from scipy.spatial.distance import cdist
                pairwise_distances = cdist(coor.values, coor.values)
                # Set diagonal to infinity to exclude self-connections
                np.fill_diagonal(pairwise_distances, np.inf)
                self.sigma = np.median(pairwise_distances[pairwise_distances > 0])

            # Compute all pairwise distances
            from scipy.spatial.distance import cdist
            distances_matrix = cdist(coor.values, coor.values)

            # Apply Gaussian kernel
            weights = np.exp(-distances_matrix**2 / (2 * self.sigma**2))

            # Set diagonal to 0 to avoid self-loops
            np.fill_diagonal(weights, 0)

            # Get non-zero connections (threshold to reduce density)
            threshold = 1e-5
            mask = (weights > threshold) & (weights > 0)

            cell1_ids, cell2_ids = np.where(mask)
            distances = distances_matrix[cell1_ids, cell2_ids]

        elif self.model == 'MNN':
            # Mutual Nearest Neighbor approach - only connect if both are among each other's k nearest
            from scipy.sparse import csr_matrix
            from sklearn.neighbors import NearestNeighbors

            # Find k nearest neighbors for each cell
            nbrs = NearestNeighbors(n_neighbors=self.k_cutoff + 1, algorithm='ball_tree').fit(coor)
            distances, indices = nbrs.kneighbors(coor)

            # Create sparse matrix representation of KNN graph
            row_idx = np.repeat(np.arange(n_cells), self.k_cutoff)
            col_idx = indices[:, 1:].flatten()  # Exclude self-connections
            data = np.ones(len(row_idx))

            knn_graph = csr_matrix((data, (row_idx, col_idx)), shape=(n_cells, n_cells))

            # Find mutual nearest neighbors: if i is in j's kNN and j is in i's kNN
            mnn_graph = knn_graph.multiply(knn_graph.T)

            # Extract edges from MNN graph
            cell1_ids, cell2_ids = mnn_graph.nonzero()
            distances = np.sqrt(np.sum((coor.iloc[cell1_ids].values - coor.iloc[cell2_ids].values)**2, axis=1))

        else:
            raise ValueError(f"Unknown model: {self.model}")

        # Create DataFrame with consistent structure
        KNN_df = pd.DataFrame({'Cell1': cell1_ids, 'Cell2': cell2_ids, 'Distance': distances})

        # Add weights if using KNN with Gaussian weights
        if self.model == 'KNN' and self.knn_weighted:
            weights = np.exp(-distances**2 / (2 * (np.std(distances) if len(distances) > 1 else 1.0)**2))
            KNN_df['Weight'] = weights

        # Map cell IDs back to original names
        id_cell_trans = dict(zip(range(n_cells), np.array(coor.index)))
        KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
        KNN_df['Cell2'] = KNN_df['Cell2'].map(id_cell_trans)

        adata.uns[self.spatial_uns] = KNN_df
        return data

    def _create_knn_graph(self, coor, n_cells):
        """Helper method to create KNN graph for fallback cases."""
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=min(10, n_cells)).fit(coor)
        distances, indices = nbrs.kneighbors(coor)

        indices = indices[:, 1:]  # Remove self-loops
        distances = distances[:, 1:]

        cell1_ids = np.repeat(np.arange(n_cells), indices.shape[1])
        cell2_ids = indices.flatten()
        distances = distances.flatten()

        return cell1_ids, cell2_ids, distances


# EVOLVE-BLOCK-END


def get_preprocessing_pipeline(log_level: LogLevel = "INFO"):
    transforms = []
    transforms.append(PrefilterGenes(min_cells=3))
    transforms.append(HighlyVariableGenesRawCount(n_top_genes=1000))
    transforms.append(NormalizeTotalLog1P())
    transforms.append(CalSpatialNet(rad_cutoff=150))
    transforms.append(SetConfig({"label_channel": "label", "label_channel_type": "obs"}))
    return Compose(*transforms, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=0, help="Number of clusters.")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--method", type=str, default="louvain", choices=["kmeans", "louvain"],
                        help="Clustering method.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0').")
    parser.add_argument("--obs_nums", type=int, default=10000)
    args = parser.parse_args()

    inner_scores = []
    scores = []
    times = []  # 新增：用于记录每次运行的时间

    for seed in range(args.seed, args.seed + args.num_runs):
        start_time = time.time()  # 新增：记录单次循环的开始时间

        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SpaGRA(k=args.k, n_epochs=args.n_epochs, method=args.method, random_seed=seed)
        preprocessing_pipeline = get_preprocessing_pipeline()

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data, args.obs_nums)
        preprocessing_pipeline(data)
        # Fit the model and evaluate
        model.fit(data.data)
        x, y = data.get_data(return_type="default")
        x = x.toarray()
        score = model.score(None, y.values)
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

        print(f"ARI: {score:.4f}, time: {run_time:.2f}s")  # 修改：打印单次得分与时间

    # 删除了原代码中这里重复的 print(f"ARI: {score:.4f}")
    print(f"spaGRA {args.sample_number}:")
    # 修改：加入 times 列表，以供 evaluator 捕获
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")  # 新增：输出平均运行时间
""" To reproduce SpaGRA on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spaGRA.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python spaGRA.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python spaGRA.py --sample_number 151507
"""
