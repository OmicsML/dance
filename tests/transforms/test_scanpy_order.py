import numpy as np
import pytest
from anndata import AnnData

from dance.data import Data
from dance.transforms import FilterGenesScanpyOrder

SEED = 123


@pytest.fixture
def toy_data():
    x = np.random.default_rng(SEED).random((50, 30)) * 100
    adata = AnnData(X=x, dtype=np.int32)
    data = Data(adata.copy())
    return adata, data


def test_sc3_feature(toy_data):
    adata, data = toy_data
    filterGenesScanpy = FilterGenesScanpyOrder(order_index=0, min_counts=1, min_cells=1, max_counts=3000, max_cells=20)
    filterGenesScanpy(data)
    sc3_feature = data.get_feature(return_type="numpy")
    assert sc3_feature.shape[0] == data.shape[0]
