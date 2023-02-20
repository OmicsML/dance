"""Reimplementation of stLearn.

Extended from https://github.com/BiomedicalMachineLearning/stLearn

Reference
----------
Pham, Duy, et al. "stLearn: integrating spatial location, tissue morphology and gene expression to find cell types,
cell-cell interactions and spatial trajectories within undissociated tissues." BioRxiv (2020).

"""
import scanpy as sc
from sklearn.cluster import KMeans

from dance.modules.base import BaseClusteringMethod
from dance.modules.spatial.spatial_domain.louvain import Louvain
from dance.transforms import AnnDataTransform, CellPCA, Compose, MorphologyFeature, SetConfig, SMEFeature
from dance.transforms.graph import NeighborGraph, SMEGraph
from dance.typing import LogLevel, Optional


class StKmeans(BaseClusteringMethod):
    """StKmeans class.

    Parameters
    ----------
    n_clusters
        The number of clusters to form as well as the number of centroids to generate.
    init
        Method for initialization: {‘k-means++’, ‘random’}.
    n_init
        Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of inertia.
    max_iter
        Maximum number of iterations of the k-means algorithm for a single run.
    tol
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two
        consecutive iterations to declare convergence.
    algorithm
        {“lloyd”, “elkan”, “auto”, “full”}, default is "auto".
    verbose
        Verbosity.
    random_state
        Determines random number generation for centroid initialization.
    use_data
        Default "X_pca".
    key_added
        Default "X_pca_kmeans".

    """

    def __init__(
        self,
        n_clusters: int = 19,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "auto",
        verbose: bool = False,
        random_state: Optional[int] = None,
        use_data: str = "X_pca",
        key_added: str = "X_pca_kmeans",
    ):
        self.use_data = use_data
        self.key_added = key_added
        self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                            algorithm=algorithm, verbose=verbose, random_state=random_state)

    @staticmethod
    def preprocessing_pipeline(morph_feat_dim: int = 50, sme_feat_dim: int = 50, pca_feat_dim: int = 10,
                               device: str = "cpu", log_level: LogLevel = "INFO"):
        return Compose(
            AnnDataTransform(sc.pp.filter_genes, min_cells=1),
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            MorphologyFeature(n_components=morph_feat_dim, device=device),
            CellPCA(n_components=pca_feat_dim),
            SMEGraph(),
            SMEFeature(n_components=sme_feat_dim),
            SetConfig({
                "feature_channel": "SMEFeature",
                "feature_channel_type": "obsm",
                "label_channel": "label",
                "label_channel_type": "obs",
            }),
            log_level=log_level,
        )

    def fit(self, x):
        """Fit function for model training.

        Parameters
        ----------
        x
            Input cell feature.

        """
        self.model.fit(x)

    def predict(self, x=None):
        """Prediction function."""
        pred = self.model.labels_
        return pred


class StLouvain(BaseClusteringMethod):
    """StLouvain class.

    Parameters
    ----------
    resolution
        Resolution parameter.

    """

    def __init__(self, resolution: float = 1):
        self.model = Louvain(resolution)

    @staticmethod
    def preprocessing_pipeline(morph_feat_dim: int = 50, sme_feat_dim: int = 50, pca_feat_dim: int = 10,
                               nbrs_pcs: int = 10, n_neighbors: int = 10, device: str = "cpu",
                               log_level: LogLevel = "INFO"):
        return Compose(
            AnnDataTransform(sc.pp.filter_genes, min_cells=1),
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            MorphologyFeature(n_components=morph_feat_dim, device=device),
            CellPCA(n_components=pca_feat_dim),
            SMEGraph(),
            SMEFeature(n_components=sme_feat_dim),
            NeighborGraph(n_neighbors=n_neighbors, n_pcs=nbrs_pcs, channel="SMEFeature"),
            SetConfig({
                "feature_channel": "NeighborGraph",
                "feature_channel_type": "obsp",
                "label_channel": "label",
                "label_channel_type": "obs",
            }),
            log_level=log_level,
        )

    def fit(self, adj, partition=None, weight="weight", randomize=None, random_state=None):
        """Fit function for model training.

        Parameters
        ----------
        adj
            Adjacent matrix.
        partition : dict
            A dictionary where keys are graph nodes and values the part the node
            belongs to
        weight : str,
            The key in graph to use as weight. Default to "weight"
        resolution : float
            Resolution.
        randomize : boolean
            Will randomize the node evaluation order and the community evaluation
            order to get different partitions at each call
        random_state : int, RandomState instance or None
            If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
            is the random number generator; If None, the random number generator is the RandomState instance used by
            :func:`numpy.random`.

        """
        self.model.fit(adj, partition, weight, randomize, random_state)

    def predict(self, x=None):
        """Prediction function."""
        pred = self.model.predict()
        return pred
