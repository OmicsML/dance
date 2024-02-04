import numpy as np
import pytest
from anndata import AnnData

from dance.data import Data
from dance.transforms import (
    FilterCellsScanpyOrder,
    FilterGenesScanpyOrder,
    HighlyVariableGenesLogarithmizedByMeanAndDisp,
    HighlyVariableGenesLogarithmizedByTopGenes,
    HighlyVariableGenesRawCount,
)

SEED = 123


@pytest.fixture
def toy_data():
    x = np.random.default_rng(SEED).random((50, 30)) * 10
    adata = AnnData(X=x)
    data = Data(adata.copy())
    return adata, data


@pytest.mark.parametrize(
    'order',
    [
        ['min_counts', 'min_cells', 'max_counts', 'max_cells'],
        ['max_counts', 'min_cells', 'min_counts'],
        ['min_cells', 'min_counts'],
        ['min_counts'],
        [],
    ],
)
def test_filter_genes_scanpy_order(toy_data, order):
    adata, data = toy_data
    filterGenesScanpy = FilterGenesScanpyOrder(order=order, min_counts=1, min_cells=1, max_counts=3000, max_cells=20)
    filterGenesScanpy(data)
    X = data.get_feature(return_type="numpy")
    assert X.shape[0] == data.shape[0]


@pytest.mark.parametrize(
    'order',
    [
        ['min_counts', 'min_genes', 'max_counts', 'max_genes'],
        ['max_counts', 'min_genes', 'min_counts'],
        ['min_genes', 'min_counts'],
        ['min_counts'],
        [],
    ],
)
def test_filter_cells_scanpy_order(toy_data, order):
    adata, data = toy_data
    filterCellsScanpy = FilterCellsScanpyOrder(order=order, min_counts=1, min_genes=1, max_counts=3000, max_genes=20)
    filterCellsScanpy(data)
    X = data.get_feature(return_type="numpy")
    assert X.shape[1] == data.shape[1]


def test_hvg(subtests):
    adata = AnnData(X=np.log1p(np.array(np.arange(1500)).reshape(50, 30)))
    data = Data(adata.copy())

    with subtests.test("HighlyVariableGenesRawCount"):
        hvg = HighlyVariableGenesRawCount(n_top_genes=20)
        hvg(data)
        assert adata.X.shape[0] == data.data.X.shape[0]

    with subtests.test("HighlyVariableGenesLogarithmizedByTopGenes"):
        hvg = HighlyVariableGenesLogarithmizedByTopGenes(n_top_genes=20)
        hvg(data)
        assert adata.X.shape[0] == data.data.X.shape[0]

    with subtests.test("HighlyVariableGenesLogarithmizedByMeanAndDisp"):
        hvg = HighlyVariableGenesLogarithmizedByMeanAndDisp()
        hvg(data)
        assert adata.X.shape[0] == data.data.X.shape[0]
