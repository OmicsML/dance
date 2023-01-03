import numba
import numpy as np
import torch

from dance.typing import NormMode


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


@numba.njit("f4[:,:](f4[:,:], u4)", parallel=True, nogil=True)
def pairwise_distance(x, dist_func_id=0):
    if dist_func_id == 0:  # Euclidean distance
        dist = euclidean_distance
    else:
        raise ValueError("Unknown distance function ID")

    n = x.shape[0]
    mat = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            mat[i][j] = dist(x[i], x[j])
    return mat
