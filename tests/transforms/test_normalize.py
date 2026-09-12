import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from dance.data import Data
from dance.transforms import ColumnSumNormalize, Log1P, NormalizeTotal, ScTransform


def test_normalize_total(subtests, assert_ary_isclose):
    adata = AnnData(X=np.array([[1, 1, 1], [1, 1, 1], [3, 0, 0]]))
    data = Data(adata.copy())

    with subtests.test("max_fraction is less than 1.0"):
        normalizeTotal = NormalizeTotal(max_fraction=0.99, target_sum=30)
        normalizeTotal(data)
        # ans = np.array([[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [90.0, 0.0, 0.0]])
        # NOTE: Divide by zero bug patched in scanpy 1.10.1. See
        # https://github.com/scverse/scanpy/pull/2856, more specifically the
        # allow_divide_by_zero option in the axis_mul_or_truediv function.
        # In this test case, the third cell has count zero since the first
        # gene is left out due to max_fraction. Consequently, the last cell
        # should be unmodified after applying normalize_total (count replaced
        # with one).
        ans = np.array([[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [3.0, 0.0, 0.0]])
        assert_ary_isclose(data.data.X, ans)

    with subtests.test("max_fraction is equal to 1.0"):
        normalizeTotal = NormalizeTotal(max_fraction=1.0, target_sum=30)
        normalizeTotal(data)
        ans = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [30.0, 0.0, 0.0]])
        assert_ary_isclose(data.data.X, ans)


def test_log1p(assert_ary_isclose):
    x = np.array([[1, 1, 1], [1, 1, 1], [3, 0, 0]])
    adata = AnnData(X=x.copy())
    data = Data(adata.copy())

    log1p = Log1P()
    log1p(data)

    ans = np.log1p(x)
    assert data.data.X.shape == adata.X.shape
    assert_ary_isclose(data.data.X, ans)


def test_column_sum_normalize_preserves_sparse():
    dense = np.array([[0, 1, 2], [3, 0, 4], [0, 5, 0]], dtype=np.float64)
    dense_data = Data(AnnData(X=dense.copy()))
    sparse_data = Data(AnnData(X=sp.csr_matrix(dense)))

    ColumnSumNormalize(axis=0)(dense_data)
    ColumnSumNormalize(axis=0, preserve_sparse=True)(sparse_data)

    assert sp.isspmatrix_csr(sparse_data.data.X)
    np.testing.assert_allclose(sparse_data.data.X.toarray(), dense_data.data.X)


def test_normalize_total_preserves_sparse():
    dense = np.array([[1, 1, 1], [1, 1, 1], [3, 0, 0]], dtype=np.float64)
    dense_data = Data(AnnData(X=dense.copy()))
    sparse_data = Data(AnnData(X=sp.csr_matrix(dense)))

    NormalizeTotal(max_fraction=1.0, target_sum=30)(dense_data)
    NormalizeTotal(max_fraction=1.0, target_sum=30, preserve_sparse=True)(sparse_data)

    assert sp.isspmatrix_csr(sparse_data.data.X)
    np.testing.assert_allclose(sparse_data.data.X.toarray(), dense_data.data.X)


def test_sc_transform_preserves_sparse_and_values():
    rng = np.random.default_rng(0)
    dense = rng.poisson(np.linspace(1, 8, 20), size=(50, 20)).astype(np.float64)
    obs_names = [f"cell-{i}" for i in range(dense.shape[0])]
    var_names = [f"gene-{i}" for i in range(dense.shape[1])]
    dense_data = Data(AnnData(X=dense.copy(), obs={"obs_names": obs_names}, var={"var_names": var_names}))
    sparse_data = Data(AnnData(X=sp.csr_matrix(dense), obs={"obs_names": obs_names}, var={"var_names": var_names}))
    dense_kwargs = {"min_cells": 1, "n_genes": None, "n_cells": None, "processes_num": 1}
    sparse_kwargs = {**dense_kwargs, "preserve_sparse": True}

    ScTransform(**dense_kwargs)(dense_data)
    ScTransform(**sparse_kwargs)(sparse_data)

    assert sp.isspmatrix_csr(sparse_data.data.X)
    assert sp.isspmatrix_csr(sparse_data.data.raw.X)
    np.testing.assert_allclose(sparse_data.data.X.toarray(), dense_data.data.X, rtol=1e-12, atol=1e-12)
