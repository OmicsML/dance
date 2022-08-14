"""Reimplementation of stLearn.

Extended from https://github.com/BiomedicalMachineLearning/stLearn

Reference
----------
Pham, Duy, et al. "stLearn: integrating spatial location, tissue morphology and gene expression to find cell types,
cell-cell interactions and spatial trajectories within undissociated tissues." BioRxiv (2020).

"""

# load kmeans from sklearn
from sklearn.cluster import KMeans

from .louvain import Louvain


# kmeans for adata
def stKmeans(adata, n_clusters=19, init="k-means++", n_init=10, max_iter=300, tol=1e-4, algorithm='auto', verbose=False,
             random_state=None, use_data='X_pca', key_added="X_pca_kmeans"):
    # kmeans for gene expression data
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, algorithm=algorithm,
                    verbose=verbose, random_state=random_state).fit(adata.obsm[use_data])
    adata.obs[key_added] = kmeans.labels_
    return adata


def stPrepare(adata):
    adata.obs['imagerow'] = adata.obs['x_pixel']
    adata.obs['imagecol'] = adata.obs['y_pixel']
    adata.obs['array_row'] = adata.obs['x']
    adata.obs['array_col'] = adata.obs['y']


class StKmeans:
    """StKmeans class.

    Parameters
    ----------
    n_clusters : int optional
        the number of clusters to form as well as the number of centroids to generate.
    init : str optional
        method for initialization: {‘k-means++’, ‘random’}.
    n_init : int optional
        number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of inertia.
    max_iter : int optional
        maximum number of iterations of the k-means algorithm for a single run.
    tol : float optional
        relative tolerance with regards to Frobenius norm of the difference in
        the cluster centers of two consecutive iterations to declare convergence.
    algorithm : str optional
        {“lloyd”, “elkan”, “auto”, “full”}, by default "auto"
    verbose : bool optional
        verbosity mode.
    random_state : int optional
        determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic.
    use_data : str optional
        by default 'X_pca'.
    key_added : str optional
        by default 'X_pca_kmeans'.

    """

    def __init__(self, n_clusters=19, init="k-means++", n_init=10, max_iter=300, tol=1e-4, algorithm='auto',
                 verbose=False, random_state=None, use_data='X_pca', key_added="X_pca_kmeans"):
        self.use_data = use_data
        self.key_added = key_added
        self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                            algorithm=algorithm, verbose=verbose, random_state=random_state)

    def fit(self, adata):
        """fit function for model training.

        Parameters
        ----------
        adata :
            input data.

        Returns
        -------
        None.

        """
        self.model.fit(adata.obsm[self.use_data])
        adata.obs[self.key_added] = self.model.labels_

    def predict(self):
        """prediction function.

        Parameters
        ----------

        Returns
        -------
        self.model.labels_ :
            predicted label.

        """
        self.predict = self.model.labels_
        self.y_pred = self.predict
        return self.predict

    def score(self, y_true):
        """score function.

        Parameters
        ----------
        adata :
            input data.

        Returns
        -------
        self.score :
            score.

        """
        from sklearn.metrics.cluster import adjusted_rand_score
        score = adjusted_rand_score(y_true, self.y_pred)
        print("ARI {}".format(adjusted_rand_score(y_true, self.y_pred)))
        return score


class StLouvain:
    """StLouvain class."""

    def __init__(self):
        self.model = Louvain()

    def fit(self, adata, adj, partition=None, weight='weight', resolution=1., randomize=None, random_state=None):
        """fit function for model training.

        Parameters
        ----------
        adata :
            input data.
        adj :
            adjacent matrix.
        partition : dict optional
            a dictionary where keys are graph nodes and values the part the node
            belongs to
        weight : str, optional
            the key in graph to use as weight. Default to 'weight'
        resolution : float optional
            resolution.
        randomize : boolean, optional
            Will randomize the node evaluation order and the community evaluation
            order to get different partitions at each call
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        Returns
        -------
        None.

        """
        self.data = adata
        self.model.fit(adata, adj, partition, weight, resolution, randomize, random_state)

    def predict(self):
        """prediction function.

        Parameters
        ----------

        Returns
        -------
        self.y_pred :
            predicted label.

        """

        self.y_pred = self.model.predict()
        self.y_pred = [self.y_pred[i] for i in range(len(self.y_pred))]
        self.data.obs['predict'] = self.y_pred
        return self.y_pred

    def score(self, y_true):
        """score function.

        Parameters
        ----------
        adata :
            input data.

        Returns
        -------
        self.score :
            score.

        """
        self.data.obs['ground'] = y_true
        tempdata = self.data.obs.dropna()
        from sklearn.metrics.cluster import adjusted_rand_score
        score = adjusted_rand_score(tempdata['ground'], tempdata['predict'])
        print("ARI {}".format(adjusted_rand_score(tempdata['ground'], tempdata['predict'])))
        return score
