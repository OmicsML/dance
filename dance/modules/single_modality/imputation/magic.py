import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, find, issparse
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from dance.modules.base import BaseRegressionMethod
from dance.transforms.cell_feature import CellPCA
from dance.transforms.filter import FilterCellsScanpy, FilterGenesScanpy
from dance.transforms.interface import AnnDataTransform
from dance.transforms.mask import CellwiseMaskData
from dance.transforms.misc import Compose, SaveRaw, SetConfig
from dance.typing import LogLevel


def magic(data, pca_projected_data, t=6, k=30, ka=10, epsilon=1, rescale=99):

    #run diffusion maps to get markov matrix
    L = compute_markov(pca_projected_data, k=k, epsilon=epsilon, distance_metric='euclidean', ka=ka)

    #remove tsne kernel for now
    # else:
    #     distances = pairwise_distances(pca_projected_data, squared=True)
    #     if k_knn > 0:
    #         neighbors_nn = np.argsort(distances, axis=0)[:, :k_knn]
    #         P = _joint_probabilities_nn(distances, neighbors_nn, perplexity, 1)
    #     else:
    #         P = _joint_probabilities(distances, perplexity, 1)
    #     P = squareform(P)

    #     #markov normalize P
    #     L = np.divide(P, np.sum(P, axis=1))

    #get imputed data matrix -- by default use data_norm but give user option to pick
    new_data, L_t = impute_fast(data, L, t, rescale_percent=rescale)

    return new_data


def impute_fast(data, L, t, rescale_percent=0, L_t=None, tprev=None):

    #convert L to full matrix
    if issparse(L):
        L = L.todense()

    #L^t
    print('MAGIC: L_t = L^t')
    if L_t == None:
        L_t = np.linalg.matrix_power(L, t)
    else:
        L_t = np.dot(L_t, np.linalg.matrix_power(L, t - tprev))

    print('MAGIC: data_new = L_t * data')
    data_new = np.array(np.dot(L_t, data))

    #rescale data
    if rescale_percent != 0:
        if len(np.where(data_new < 0)[0]) > 0:
            print('Rescaling should not be performed on log-transformed '
                  '(or other negative) values. Imputed data returned unscaled.')
            return data_new, L_t

        M99 = np.percentile(data, rescale_percent, axis=0)
        M100 = data.max(axis=0)
        indices = np.where(M99 == 0)[0]
        M99[indices] = M100[indices]
        M99_new = np.percentile(data_new, rescale_percent, axis=0)
        M100_new = data_new.max(axis=0)
        indices = np.where(M99_new == 0)[0]
        M99_new[indices] = M100_new[indices]
        max_ratio = np.divide(M99, M99_new)
        data_new = np.multiply(data_new, np.tile(max_ratio, (len(data), 1)))

    return data_new, L_t


def compute_markov(data, k=10, epsilon=1, distance_metric='euclidean', ka=0):

    N = data.shape[0]

    # Nearest neighbors
    print('Computing distances')
    nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(data)
    distances, indices = nbrs.kneighbors(data)

    if ka > 0:
        print('Autotuning distances')
        for j in reversed(range(N)):
            temp = sorted(distances[j])
            lMaxTempIdxs = min(ka, len(temp))
            if lMaxTempIdxs == 0 or temp[lMaxTempIdxs] == 0:
                distances[j] = 0
            else:
                distances[j] = np.divide(distances[j], temp[lMaxTempIdxs])

    # Adjacency matrix
    print('Computing kernel')
    rows = np.zeros(N * k, dtype=np.int32)
    cols = np.zeros(N * k, dtype=np.int32)
    dists = np.zeros(N * k)
    location = 0
    for i in range(N):
        inds = range(location, location + k)
        rows[inds] = indices[i, :]
        cols[inds] = i
        dists[inds] = distances[i, :]
        location += k
    if epsilon > 0:
        W = csr_matrix((dists, (rows, cols)), shape=[N, N])
    else:
        W = csr_matrix((np.ones(dists.shape), (rows, cols)), shape=[N, N])

    # Symmetrize W
    W = W + W.T

    if epsilon > 0:
        # Convert to affinity (with selfloops)
        rows, cols, dists = find(W)
        rows = np.append(rows, range(N))
        cols = np.append(cols, range(N))
        dists = np.append(dists / (epsilon**2), np.zeros(N))
        W = csr_matrix((np.exp(-dists), (rows, cols)), shape=[N, N])

    # Create D
    D = np.ravel(W.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]

    #markov normalization
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(W)

    return T


def optimal_t(data, th=0.001):
    S = np.linalg.svd(data)
    S = np.power(S, 2)
    nse = np.zeros(32)

    for t in range(32):
        S_t = np.power(S, t)
        P = np.divide(S_t, np.sum(S_t, axis=0))
        nse[t] = np.sum(P[np.where(P > th)[0]])


class MAGIC(nn.Module, BaseRegressionMethod):

    def __init__(self, t=6, k=30, ka=10, epsilon=1, rescale=99, gpu=-1):
        super().__init__()
        self.t = t
        self.k = k
        self.ka = ka
        self.epsilon = epsilon
        self.rescale = rescale
        self.device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')

    def fit(self, **kwargs):
        pass

    def predict(self, X_test, X_test_pca):
        y = magic(X_test, X_test_pca, t=self.t, k=self.k, ka=self.ka, epsilon=self.epsilon, rescale=self.rescale)
        return torch.tensor(y).float().to(self.device)

    @staticmethod
    def preprocessing_pipeline(min_cells: float = 0.1, dim=20, mask: bool = True, distr: str = "exp",
                               mask_rate: float = 0.1, seed: int = 1, log_level: LogLevel = "INFO"):
        transforms = [
            FilterGenesScanpy(min_cells=min_cells),
            FilterCellsScanpy(min_counts=1),
            SaveRaw(),
            AnnDataTransform(sc.pp.log1p),
            CellPCA(n_components=dim),
        ]
        if mask:
            transforms.extend([
                CellwiseMaskData(distr=distr, mask_rate=mask_rate, seed=seed),
                SetConfig({
                    "feature_channel": [None, None, "train_mask", "CellPCA"],
                    "feature_channel_type": ["X", "raw_X", "layers", "obsm"],
                    "label_channel": [None, None],
                    "label_channel_type": ["X", "raw_X"],
                })
            ])
        else:
            transforms.extend([
                SetConfig({
                    "feature_channel": [None, None],
                    "feature_channel_type": ["X", "raw_X"],
                    "label_channel": [None, None],
                    "label_channel_type": ["X", "raw_X"],
                })
            ])

        return Compose(*transforms, log_level=log_level)

    def score(self, true_expr, imputed_expr, mask=None, metric="MSE", test_idx=None):
        """Scoring function of model.

        Parameters
        ----------
        true_expr :
            True underlying expression values
        imputed_expr :
            Imputed expression values
        test_idx :
            index of testing cells
        metric :
            Choice of scoring metric - 'RMSE' or 'ARI'

        Returns
        -------
        score :
            evaluation score

        """
        allowd_metrics = {"RMSE", "PCC", "MRE"}
        if metric not in allowd_metrics:
            raise ValueError("scoring metric %r." % allowd_metrics)

        if test_idx is None:
            test_idx = range(len(true_expr))
        true_target = true_expr.to(self.device)
        imputed_target = imputed_expr.to(self.device)
        if mask is not None:  # and metric == 'MSE':
            # true_target = true_target[~mask[test_idx]]
            # imputed_target = imputed_target[~mask[test_idx]]
            imputed_target[mask[test_idx]] = true_target[mask[test_idx]]
        if metric == 'RMSE':
            return np.sqrt(F.mse_loss(true_target, imputed_target).item())
        elif metric == 'PCC':
            # corr_cells = np.corrcoef(true_target.cpu(), imputed_target.cpu())
            # return corr_cells
            return np.corrcoef(true_target.cpu()[~mask[test_idx]], imputed_target.cpu()[~mask[test_idx]])[0, 1]
        elif metric == "MRE":
            actual = true_target.cpu()[~mask[test_idx]]
            predicted = imputed_target.cpu()[~mask[test_idx]]
            abs_error = torch.abs(predicted - actual)
            abs_actual = torch.abs(actual)
            abs_actual[abs_actual < 1e-10] = 1e-10
            relative_error = abs_error / abs_actual
            mre = torch.mean(relative_error).item()
            return mre


# add test
