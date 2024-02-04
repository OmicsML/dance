import numpy as np
from anndata import AnnData

from dance.data import Data
from dance.transforms import Log1P, NormalizeTotal


def test_normalize_total(subtests, assert_ary_isclose):
    adata = AnnData(X=np.array([[1, 1, 1], [1, 1, 1], [3, 0, 0]]))
    data = Data(adata.copy())

    with subtests.test("max_fraction is less than 1.0"):
        normalizeTotal = NormalizeTotal(max_fraction=0.99, target_sum=30)
        normalizeTotal(data)
        ans = np.array([[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [90.0, 0.0, 0.0]])
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
