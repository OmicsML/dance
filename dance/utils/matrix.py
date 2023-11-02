import numba
import numpy as np
import torch
import scipy.stats
from dance.typing import NormMode
from collections import Counter
from math import isclose
from statistics import stdev, variance
from typing import Sequence, Iterator


def normalize(mat, *, mode: NormMode = "normalize", axis: int = 0, eps: float = -1.0):
    """Normalize a matrix.

    Parameters
    ----------
    mat
        Input matrix to be normalized, can be torch tensor or numpy array.
    mode
        Normalization mode. **normalize** means divide the values by the sum. **standardize** means center then rescale
        by standard deviation. "minmax" means rescale the values along the axis of choice between zero and one.
    axis
        Axis along which the normalization will take place.
    eps
        Denominator correction factor to prevent divide by zero error. If set to -1, then replace the zero entries with
        ones.

    """
    if isinstance(mat, torch.Tensor):
        is_torch = True
    elif not isinstance(mat, np.ndarray):
        raise TypeError(f"Invalid type for input matrix: {type(mat)}")
    else:
        is_torch = False
    opts = {"axis": axis, "keepdims": True}

    # Compute shift
    if mode == "standardize":
        shift = -mat.mean(**opts)
    elif mode == "minmax":
        min_vals = mat.min(**opts)[0] if is_torch else mat.min(**opts)
        shift = -min_vals
    else:
        shift = 0

    # Compute rescaling factor
    if mode == "normalize":
        denom = mat.sum(**opts)
    elif mode == "standardize":
        denom = mat.std(**opts, unbiased=False) if is_torch else mat.std(**opts)
    elif mode == "minmax":
        max_vals = mat.max(**opts)[0] if is_torch else mat.max(**opts)
        denom = max_vals - min_vals
    elif mode == "l2":
        denom = (mat**2).sum(**opts)**0.5
    else:
        denom = None

    # Correct denominator to prevent divide by zero error
    if denom is None:
        denom = 1
    elif eps == -1:
        denom[denom == 0] = 1
    elif eps > 0:
        denom += eps
    else:
        raise ValueError(f"Invalid {eps=!r}. Must be positive or -1, the later set zero entries to one.")

    norm_mat = (mat + shift) / denom

    return norm_mat


@numba.njit("f4(f4[:], f4[:])")
def euclidean_distance(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i])**2
    return np.sqrt(sum)
    
@numba.njit("f4(f4[:], f4[:])")
def pearson_distance(a, b):
    a_avg = np.sum(a) / len(a)
    b_avg = np.sum(b) / len(b)
    cov_ab1 = [x-a_avg for x in a]
    cov_ab2 = [y-b_avg for y in b]
    cov_ab  = np.sum(np.array([cov_ab1[i]*cov_ab2[i] for i in range(len(cov_ab1))]))
    sq = (np.sum(np.array([(x - a_avg) ** 2 for x in a])) * np.sum(np.array([(x - b_avg) ** 2 for x in b]))) ** 0.5
    corr_factor = cov_ab / sq
    return 1-corr_factor


def rank_series(values: Sequence[float]) -> Iterator[float]:
    rank_by_value = {}
    index = 0
    for x, n in sorted(Counter(values).items()):
        rank_by_value[x] = index + (1 + n)/2
        index += n
    for x in values:
        yield rank_by_value[x]


@numba.njit("f4(f4[:], f4[:])")
def spearman_distance(x, y):
    """
    The Spearman rank correlation is used to evaluate if the relationship between two variables,
    X and Y is monotonic. The rank correlation measures how closely related the ordering of one
    variable to the other variable, with no regard to the actual values of the variables.
    """
    if len(x) != len(y):
        raise ValueError(f'X length {len(x)} does not match Y length {len(y)}')
    x_ranks = np.argsort(np.argsort(x))+1.0
    y_ranks = np.argsort(np.argsort(y))+1.0
    covxy = np.cov(x_ranks, y_ranks)
    stdx = np.sqrt(((x_ranks- np.mean(x_ranks)) ** 2).sum() / (x_ranks.size - 1))
    stdy = np.sqrt(((y_ranks- np.mean(y_ranks)) ** 2).sum() / (y_ranks.size - 1))
    r = covxy/stdx/stdy
    return 1-r[0,1]


@numba.njit("f4[:,:](f4[:,:], u4)", parallel=True, nogil=True)
def pairwise_distance(x, dist_func_id=0):
    if dist_func_id == 0:  # Euclidean distance
        dist = euclidean_distance
    elif dist_func_id==1:
        dist=pearson_distance
    elif dist_func_id==2:
        dist=spearman_distance
    else:
        raise ValueError("Unknown distance function ID")

    n = x.shape[0]
    mat = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            mat[i][j] = dist(x[i], x[j])
    return mat
