import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

from dance.data import Data
from dance.transforms.interface import AnnDataTransform

SEED = 123


@pytest.fixture
def toy_data():
    x = np.random.default_rng(SEED).random((5, 3))
    adata = AnnData(X=x, dtype=np.float32)
    data = Data(adata.copy())
    return adata, data


def test_anndata_transform_from_func(toy_data):
    adata, data = toy_data

    sc.pp.normalize_total(adata, target_sum=100)  # transform via sc.pp
    AnnDataTransform(sc.pp.normalize_total, target_sum=100)(data)  # transform via interface
    assert adata.X.tolist() == data.data.X.tolist()

    sc.pp.log1p(adata)  # transform via sc.pp
    AnnDataTransform(sc.pp.log1p)(data)  # transform via interface
    assert adata.X.tolist() == data.data.X.tolist()

    with pytest.raises(TypeError):
        AnnDataTransform(sc.pp)  # must be callable


def test_anndata_transform_from_str(toy_data):
    adata, data = toy_data

    sc.pp.normalize_total(adata, target_sum=100)  # transform via sc.pp
    AnnDataTransform("scanpy.pp.normalize_total", target_sum=100)(data)  # transform via interface resolved from str
    assert adata.X.tolist() == data.data.X.tolist()

    sc.pp.log1p(adata)  # transform via sc.pp
    AnnDataTransform("scanpy.pp.log1p")(data)  # transform via interface resolved from str
    assert adata.X.tolist() == data.data.X.tolist()

    with pytest.raises(TypeError):
        AnnDataTransform("scanpy.pp")  # must be callable

    with pytest.raises(AttributeError):
        AnnDataTransform("scanpy.pp.dosenot_exist")  # must be resolvable
