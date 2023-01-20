"""Reimplementation of stLearn.

Extended from https://github.com/BiomedicalMachineLearning/stLearn

Reference
----------
Pham, Duy, et al. "stLearn: integrating spatial location, tissue morphology and gene expression to find cell types,
cell-cell interactions and spatial trajectories within undissociated tissues." BioRxiv (2020).

"""

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

from dance.modules.spatial.spatial_domain.louvain import Louvain


class StKmeans:
    """StKmeans class."""

    def __init__(self, n_clusters=19, init="k-means++", n_init=10, max_iter=300, tol=1e-4, algorithm="auto",
                 verbose=False, random_state=None, use_data="X_pca", key_added="X_pca_kmeans"):
        """Initialize StKMeans.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        init : str
            Method for initialization: {‘k-means++’, ‘random’}.
        n_init : int
            Number of time the k-means algorithm will be run with different centroid seeds.
            The final results will be the best output of n_init consecutive runs in terms of inertia.
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run.
        tol : float
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two
            consecutive iterations to declare convergence.
        algorithm : str
            {“lloyd”, “elkan”, “auto”, “full”}, default is "auto".
        verbose : bool
            Verbosity.
        random_state : int
            Determines random number generation for centroid initialization.
        use_data : str
            Default "X_pca".
        key_added : str
            Default "X_pca_kmeans".

        """
        self.use_data = use_data
        self.key_added = key_added
        self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                            algorithm=algorithm, verbose=verbose, random_state=random_state)

    def fit(self, x):
        """Fit function for model training.

        Parameters
        ----------
        x
            Input cell feature.

        """
        self.model.fit(x)

    def predict(self):
        """Prediction function."""
        self.predict = self.model.labels_
        self.y_pred = self.predict
        return self.predict

    def score(self, y_true):
        """Score function.

        Parameters
        ----------
        y_true
            Cluster labels.

        Returns
        -------
        float
            Adjusted rand index score.

        """
        score = adjusted_rand_score(y_true, self.y_pred)
        return score


class StLouvain:
    """StLouvain class."""

    def __init__(self, resolution: float = 1):
        """Initialize StLouvain.

        Parameters
        ----------
        resolution : float
            Resolution parameter.

        """
        self.model = Louvain(resolution)

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
            `np.random`.

        """
        self.model.fit(adj, partition, weight, randomize, random_state)

    def predict(self):
        """Prediction function."""
        self.y_pred = self.model.predict()
        self.y_pred = [self.y_pred[i] for i in range(len(self.y_pred))]
        return self.y_pred

    def score(self, y_true):
        """Score function.

        Parameters
        ----------
        y_true
            Cluster labels.

        Returns
        -------
        float
            Adjusted rand index score.

        """
        score = adjusted_rand_score(y_true, self.y_pred)
        return score
