import numpy as np
from anndata import AnnData

from dance.data import Data
from dance.transforms import Log1P, NormalizeTotal


def test_normalize_total(subtests):
    adata = AnnData(X=np.array([[1, 1, 1], [1, 1, 1], [3, 0, 0]]))
    data = Data(adata.copy())
    with subtests.test("max_fraction is less than 1.0"):
        normalizeTotal = NormalizeTotal(max_fraction=0.99, target_sum=30)
        normalizeTotal(data)
        assert (data.data.X == np.array([[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [90.0, 0.0, 0.0]])).all()
    with subtests.test("max_fraction is equal to 1.0"):
        normalizeTotal = NormalizeTotal(max_fraction=1.0, target_sum=30)
        normalizeTotal(data)
        assert (data.data.X == np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [30.0, 0.0, 0.0]])).all()


def test_log1p():
    adata = AnnData(X=np.array([[1, 1, 1], [1, 1, 1], [3, 0, 0]]))
    data = Data(adata.copy())
    log1p = Log1P()
    log1p(data)
    assert (data.data.X == np.array([[0.6931471805599453, 0.6931471805599453, 0.6931471805599453],
                                     [0.6931471805599453, 0.6931471805599453, 0.6931471805599453],
                                     [1.3862943611198906, 0.0, 0.0]])).all()
