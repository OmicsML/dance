import numpy as np
import pytest
from anndata import AnnData

from dance.data import Data
from dance.transforms import TangramFeature

SEED = 123


@pytest.fixture
def toy_data():
    x = np.random.default_rng(SEED).random((5, 3))
    adata = AnnData(X=x, dtype=np.float32)
    data = Data(adata.copy())
    return adata, data


def test_tangram_feature(toy_data):
    adata, data = toy_data
    tangramFeature = TangramFeature()
    data = tangramFeature(data)
    tangram_feature = data.get_feature(return_type="numpy", channel="TangramFeature", channel_type="obs")
    assert np.sum(tangram_feature) == 1
