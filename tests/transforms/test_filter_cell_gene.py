import numpy as np
import pytest
import scanpy as sc
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
def test_filter_genes_scanpy_order(toy_data, order, assert_ary_isclose):
    adata, data = toy_data
    kwargs = dict(min_counts=1, min_cells=1, max_counts=3000, max_cells=20)
    filterGenesScanpy = FilterGenesScanpyOrder(order=order, **kwargs)
    filterGenesScanpy(data)

    for i in order:
        sc.pp.filter_genes(adata, **{i: kwargs[i]})
    ans = adata.X

    assert data.data.X.shape == ans.shape
    assert_ary_isclose(data.data.X, ans)


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
def test_filter_cells_scanpy_order(toy_data, order, assert_ary_isclose):
    adata, data = toy_data
    kwargs = dict(min_counts=1, min_genes=1, max_counts=3000, max_genes=20)
    filterCellsScanpy = FilterCellsScanpyOrder(order=order, **kwargs)
    filterCellsScanpy(data)

    for i in order:
        sc.pp.filter_cells(adata, **{i: kwargs[i]})
    ans = adata.X

    assert data.data.X.shape == ans.shape
    assert_ary_isclose(data.data.X, ans)

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
