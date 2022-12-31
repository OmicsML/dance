import numpy as np
import scanpy as sc
from anndata import AnnData

from dance.data import Data
from dance.transforms.interface import AnnDataTransform

SEED = 123


def test_anndata_transform():
    x = np.random.default_rng(SEED).random((5, 3))
    adata = AnnData(X=x, dtype=np.float32)
    data = Data(adata.copy())

    sc.pp.normalize_total(adata, target_sum=100)  # transform via sc.pp
    AnnDataTransform(sc.pp.normalize_total, target_sum=100)(data)  # transform via interface
    assert adata.X.tolist() == data.data.X.tolist()

    sc.pp.log1p(adata)  # transform via sc.pp
    AnnDataTransform(sc.pp.log1p)(data)  # transform via interface
    assert adata.X.tolist() == data.data.X.tolist()
