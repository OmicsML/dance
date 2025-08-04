import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from dance.transforms.base import BaseTransform
from dance.typing import Optional
from dance.utils.matrix import pairwise_distance
from dance.utils.status import experimental


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    identity = np.eye(adj_matrix.shape[0])
    return identity - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)


# @register_preprocessor("feature", "???")  # update out channel type accordingly
@experimental(reason="Need to check whether this is cell or gene feature and update out channel type accordingly. "
              "The computateion needs to be improved: use vectorized computation instead of nested loops.")
class SC3Feature(BaseTransform):
    """SC3 features via a cluster-based similarity partitioning algorithm.

    For each individual clustering result, a binary similarity matrix is constructed from the corresponding cell labels.
    If two cells belong tothe same cluster, their similarity is 1; otherwise, the similarity is 0. A consensus matrix is
    calculated by averaging all similarity matrices of individual clusterings. To reduce computational time, if the
    length of the d rangeis more than 15, a random subset of 15 values selected uniformly from the d range is used.

    Parameters
    ----------
    n_cluster
        Number of clusters for kmeans clustering.
    d
        Number of cells selected.

    References
    ---------
    https://www.nature.com/articles/nmeth.4236

    """

    def __init__(self, n_cluster: int = 3, d: Optional[int] = None, **kwargs):

        super().__init__(**kwargs)
        self.n_cluster = n_cluster
        self.choices = None
        self.d = d

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy")

        num_cells = feat.shape[0]
        if self.d is None:
            self.d = math.ceil(num_cells * 0.07) - math.floor(num_cells * 0.04)
        if self.d > 15:
            self.choices = sorted(np.random.choice(range(self.d), 15, replace=False))
        else:
            self.choices = list(range(self.d))

        y_len = feat.shape[0]
        sc3_mats = []
        for i in range(3):
            corr = torch.from_numpy(pairwise_distance(np.array(feat).astype(np.float32), dist_func_id=i))
            sc3_mat = corr.numpy()
            mat_pca = PCA(n_components=y_len)
            sc3_mats.append(mat_pca.fit_transform(sc3_mat)[:, self.choices])
            sc3_mats.append(normalized_laplacian(sc3_mat)[:, self.choices])

        sim_matrix_all = []
        for sc3_mat in sc3_mats:
            for i in range(len(self.choices)):
                sim_matrix = np.identity(y_len)
                y_pred = KMeans(n_clusters=self.n_cluster, random_state=9).fit_predict(sc3_mat[:, 0:i + 1])
                for i in range(y_len):
                    for j in range(i + 1, y_len):
                        y1 = y_pred[i]
                        y2 = y_pred[j]
                        if (y1 == y2):
                            sim_matrix[i][j] = 1
                            sim_matrix[j][i] = 1
                sim_matrix_all.append(sim_matrix)
        sim_matrix_all = np.array(sim_matrix_all)
        sim_matrix_mean = np.mean(sim_matrix_all, axis=0)

        data.data.uns[self.out] = sim_matrix_mean
        return data
