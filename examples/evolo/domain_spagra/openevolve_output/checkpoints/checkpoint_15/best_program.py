import argparse

import numpy as np

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spaGRA import SpaGRA
from dance.transforms import Compose, HighlyVariableGenesRawCount, NormalizeTotalLog1P, PrefilterGenes, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
import pandas as pd
import sklearn

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
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors. When model=='Delaunay', 
        uses Delaunay triangulation for neighbor selection.
    spatial_uns
        Key for storing spatial networks in adata.uns. Default is "Spatial_Net".
    """

    _DISPLAY_ATTRS = ("rad_cutoff", "k_cutoff", "model", "spatial_uns")

    def __init__(self, rad_cutoff=None, k_cutoff=None, model='Radius', spatial_uns="Spatial_Net",
                 out=None, log_level="WARNING"):
        super().__init__(out=out, log_level=log_level)
        self.rad_cutoff = rad_cutoff
        self.k_cutoff = k_cutoff
        self.model = model
        self.spatial_uns = spatial_uns

    def __call__(self, data):
        """Construct spatial neighbor networks."""
        adata = data.data

        print('------Calculating spatial graph...')
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']
        
        # Get coordinates as numpy array for faster computation
        coords = coor.values
        
        if self.model == 'Radius':
            # Use radius-based neighbors with vectorized operations
            nbrs = sklearn.neighbors.NearestNeighbors(radius=self.rad_cutoff).fit(coords)
            distances, indices = nbrs.radius_neighbors(coords, return_distance=True)
            
            # Vectorized creation of edge lists
            cell1_indices = np.concatenate([np.full(len(idx), i) for i, idx in enumerate(indices)])
            cell2_indices = np.concatenate(indices)
            distances = np.concatenate(distances)
            
        elif self.model == 'KNN':
            # Use KNN with vectorized operations
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k_cutoff + 1).fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            # Remove self-connections (distance = 0) and create edge lists
            cell1_indices = np.repeat(np.arange(len(indices)), self.k_cutoff + 1)
            cell2_indices = indices.flatten()
            distances = distances.flatten()
            
            # Filter out self-connections
            mask = distances > 0
            cell1_indices = cell1_indices[mask]
            cell2_indices = cell2_indices[mask]
            distances = distances[mask]
            
        elif self.model == 'Delaunay':
            # Use Delaunay triangulation for better spatial topology
            from scipy.spatial import Delaunay
            
            try:
                tri = Delaunay(coords)
                simplices = tri.simplices
                
                # Create edges from simplex vertices
                edges = []
                for simplex in simplices:
                    # Get all unique pairs from the simplex
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            edges.append((simplex[i], simplex[j]))
                
                # Remove duplicates and create arrays
                edges = list(set(edges))
                if len(edges) > 0:
                    cell1_indices, cell2_indices = zip(*edges)
                    # Calculate actual distances for weighting
                    distances = np.sqrt(np.sum((coords[list(cell1_indices)] - coords[list(cell2_indices)])**2, axis=1))
                else:
                    # Fallback if no triangulation possible
                    cell1_indices = np.array([])
                    cell2_indices = np.array([])
                    distances = np.array([])
                    
            except Exception:
                # Fallback to KNN if Delaunay fails
                nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=min(self.k_cutoff + 1, len(coords)-1)).fit(coords)
                distances, indices = nbrs.kneighbors(coords)
                cell1_indices = np.repeat(np.arange(len(indices)), self.k_cutoff + 1)
                cell2_indices = indices.flatten()
                distances = distances.flatten()
                mask = distances > 0
                cell1_indices = cell1_indices[mask]
                cell2_indices = cell2_indices[mask]
                distances = distances[mask]
                
        elif self.model == 'Gabriel':
            # Use Gabriel graph construction
            from scipy.spatial.distance import cdist
            
            # Compute pairwise distances
            dist_matrix = cdist(coords, coords)
            
            # Create adjacency matrix based on Gabriel graph criteria
            # Two points are connected if the circle with diameter equal to their distance 
            # contains no other points
            n_points = len(coords)
            adj_matrix = np.zeros((n_points, n_points), dtype=bool)
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    if i != j:
                        # Check if any point lies inside the circle defined by i and j
                        dist_ij = dist_matrix[i, j]
                        center = (coords[i] + coords[j]) / 2
                        radius = dist_ij / 2
                        
                        # Check if any other point is within this circle
                        is_gabriel = True
                        for k in range(n_points):
                            if k != i and k != j:
                                dist_to_center = np.linalg.norm(coords[k] - center)
                                if dist_to_center < radius:
                                    is_gabriel = False
                                    break
                        if is_gabriel:
                            adj_matrix[i, j] = True
                            adj_matrix[j, i] = True
            
            # Extract edges
            edges = np.where(adj_matrix)
            cell1_indices = edges[0]
            cell2_indices = edges[1]
            
            # Calculate actual distances for edges
            distances = dist_matrix[cell1_indices, cell2_indices]
            
        else:
            raise ValueError(f"Unsupported model: {self.model}")
            
        # Create final dataframe
        if len(cell1_indices) > 0:
            KNN_df = pd.DataFrame({
                'Cell1': cell1_indices,
                'Cell2': cell2_indices,
                'Distance': distances
            })
        else:
            KNN_df = pd.DataFrame(columns=['Cell1', 'Cell2', 'Distance'])
            
        Spatial_Net = KNN_df.copy()
        
        # Map back to original cell IDs
        id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
        
        adata.uns[self.spatial_uns] = Spatial_Net
        return data
# EVOLVE-BLOCK-END

def get_preprocessing_pipeline(log_level: LogLevel = "INFO"):
    transforms = []
    transforms.append(PrefilterGenes(min_cells=3))
    transforms.append(HighlyVariableGenesRawCount(n_top_genes=1000))
    transforms.append(NormalizeTotalLog1P())
    transforms.append(CalSpatialNet(rad_cutoff=150))
    transforms.append(SetConfig({"label_channel": "label",
            "label_channel_type": "obs"}))
    return Compose(*transforms, log_level=log_level)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=202, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=0, help="Number of clusters.")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--method", type=str, default="louvain", choices=["kmeans", "louvain"],
                        help="Clustering method.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0').")
    args = parser.parse_args()
    inner_scores=[]
    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SpaGRA(k=args.k, n_epochs=args.n_epochs, method=args.method, random_seed=seed)
        preprocessing_pipeline = get_preprocessing_pipeline()

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data)
        preprocessing_pipeline(data)
        # Fit the model and evaluate
        model.fit(data.data)
        x,y = data.get_data(return_type="default")
        x=x.toarray()
        score = model.score(None, y.values)
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
