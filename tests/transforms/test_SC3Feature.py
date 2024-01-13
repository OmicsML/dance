import numpy as np
import pytest
from anndata import AnnData

from dance.data import Data
from dance.transforms import SC3Feature

SEED = 123


@pytest.fixture
def toy_data():
    x = np.random.default_rng(SEED).random((5, 3))
    adata = AnnData(X=x, dtype=np.float32)
    data = Data(adata.copy())
    return adata, data


def test_sc3_feature(toy_data):
    adata, data = toy_data
    sc3feature = SC3Feature()
    data = sc3feature(data)
    sc3_feature = data.get_feature(return_type="numpy", channel="SC3Feature", channel_type="obsm")
    assert sc3_feature.shape[0] == data.shape[0]
