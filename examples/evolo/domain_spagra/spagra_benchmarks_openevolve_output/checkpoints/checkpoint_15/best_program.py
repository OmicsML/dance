import argparse

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
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff.
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors. When model=='Delaunay', uses Delaunay triangulation.
        When model=='Gaussian', uses KNN with Gaussian kernel weights.
    spatial_uns
        Key for storing spatial networks in adata.uns. Default is "Spatial_Net".
    sigma
        Standard deviation for Gaussian kernel when model=='Gaussian'. Default is 100.

    """

    _DISPLAY_ATTRS = ("rad_cutoff", "k_cutoff", "model", "spatial_uns")

    def __init__(self, rad_cutoff=None, k_cutoff=None, model='Radius', spatial_uns="Spatial_Net", out=None,
                 log_level="WARNING", sigma=100):
        super().__init__(out=out, log_level=log_level)
        self.rad_cutoff = rad_cutoff
        self.k_cutoff = k_cutoff
        self.model = model
        self.spatial_uns = spatial_uns
        self.sigma = sigma

    def __call__(self, data):
        """Construct spatial neighbor networks."""
        adata = data.data

        print('------Calculating spatial graph...')
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']

        # Convert coordinates to numpy for faster computation
        coords = coor.values

        if self.model == 'Radius':
            # Use Radius-based approach with vectorized operations
            nbrs = sklearn.neighbors.NearestNeighbors(radius=self.rad_cutoff).fit(coords)
            distances, indices = nbrs.radius_neighbors(coords, return_distance=True)

            # Vectorized construction of edge lists
            cell1_ids = np.concatenate([np.full(len(idx), i) for i, idx in enumerate(indices)])
            cell2_ids = np.concatenate(indices)
            distances = np.concatenate(distances)

            # Remove self-loops and zero distances
            mask = (cell1_ids != cell2_ids) & (distances > 0)
            cell1_ids = cell1_ids[mask]
            cell2_ids = cell2_ids[mask]
            distances = distances[mask]

        elif self.model == 'KNN':
            # Use KNN-based approach with vectorized operations
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k_cutoff + 1).fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            # Remove self-loops (first neighbor is always the point itself)
            cell1_ids = np.repeat(np.arange(coords.shape[0]), self.k_cutoff)
            cell2_ids = indices[:, 1:].flatten()  # Exclude first neighbor (self)
            distances = distances[:, 1:].flatten()  # Exclude first distance (0)

        elif self.model == 'Delaunay':
            # Use Delaunay triangulation for more biologically meaningful connections
            from scipy.spatial import Delaunay
            tri = Delaunay(coords)
            simplex_indices = tri.simplices

            # Get all edges from simplices (triangles)
            edges = []
            for simplex in simplex_indices:
                # Generate all unique pairs from the triangle vertices
                for i in range(3):
                    for j in range(i + 1, 3):
                        edges.append((simplex[i], simplex[j]))

            # Remove duplicates and self-loops
            edges = list(set(edges))
            edges = [(u, v) for u, v in edges if u != v]

            if not edges:
                raise ValueError(
                    "No edges found in Delaunay triangulation. Try increasing k_cutoff or using a different model.")

            cell1_ids, cell2_ids = zip(*edges)
            # Calculate actual distances for the edges
            distances = np.sqrt(np.sum((coords[list(cell1_ids)] - coords[list(cell2_ids)])**2, axis=1))

        elif self.model == 'Gaussian':
            # Use KNN with Gaussian kernel weights
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k_cutoff + 1).fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            # Remove self-loops and compute Gaussian weights
            cell1_ids = np.repeat(np.arange(coords.shape[0]), self.k_cutoff)
            cell2_ids = indices[:, 1:].flatten()
            distances = distances[:, 1:].flatten()

            # Apply Gaussian kernel: exp(-distance^2 / (2 * sigma^2))
            weights = np.exp(-distances**2 / (2 * self.sigma**2))

        else:
            raise ValueError(f"Unknown model '{self.model}'. Choose from 'Radius', 'KNN', 'Delaunay', 'Gaussian'")

        # Create the final DataFrame
        if self.model == 'Gaussian':
            # For Gaussian model, we don't want to filter by distance threshold like others
            df = pd.DataFrame({'Cell1': cell1_ids, 'Cell2': cell2_ids, 'Distance': distances, 'Weight': weights})
        else:
            # Filter out zero distances and self-loops
            mask = (distances > 0) & (cell1_ids != cell2_ids)
            cell1_ids = cell1_ids[mask]
            cell2_ids = cell2_ids[mask]
            distances = distances[mask]

            df = pd.DataFrame({'Cell1': cell1_ids, 'Cell2': cell2_ids, 'Distance': distances})

        # Map cell IDs back to original names
        id_cell_trans = dict(zip(range(coor.shape[0]), coor.index))
        df['Cell1'] = df['Cell1'].map(id_cell_trans)
        df['Cell2'] = df['Cell2'].map(id_cell_trans)

        adata.uns[self.spatial_uns] = df
        return data


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
    for seed in range(args.seed, args.seed + args.num_runs):
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
        print(f"ARI: {score:.4f}")

    print(f"ARI: {score:.4f}")
    print(f"spaGRA {args.sample_number}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
""" To reproduce SpaGRA on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spaGRA.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python spaGRA.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python spaGRA.py --sample_number 151507
"""
