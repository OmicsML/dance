import numpy as np
import pytest
from anndata import AnnData

from dance.data import Data
from dance.transforms import (FilterCellsScanpyOrder, FilterGenesScanpyOrder,
                              HighlyVariableGenesLogarithmizedByMeanAndDisp, HighlyVariableGenesLogarithmizedByTopGenes,
                              HighlyVariableGenesRawCount, Log1P, NormalizeTotal)

SEED = 123


@pytest.fixture
def toy_data():
    x = np.random.default_rng(SEED).random((50, 30)) * 100
    adata = AnnData(X=x)
    data = Data(adata.copy())
    return adata, data


@pytest.mark.parametrize('order',
                         [['min_counts', 'min_cells', 'max_counts', 'max_cells'],
                          ['max_counts', 'min_cells', 'min_counts'], ['min_cells', 'min_counts'], ['min_counts'], []])
def test_filter_genes_scanpy_order(toy_data, order):
    adata, data = toy_data
    filterGenesScanpy = FilterGenesScanpyOrder(order=order, min_counts=1, min_cells=1, max_counts=3000, max_cells=20)
    filterGenesScanpy(data)
    X = data.get_feature(return_type="numpy")
    assert X.shape[0] == data.shape[0]


@pytest.mark.parametrize('order',
                         [['min_counts', 'min_genes', 'max_counts', 'max_genes'],
                          ['max_counts', 'min_genes', 'min_counts'], ['min_genes', 'min_counts'], ['min_counts'], []])
def test_filter_cells_scanpy_order(toy_data, order):
    adata, data = toy_data
    filterCellsScanpy = FilterCellsScanpyOrder(order=order, min_counts=1, min_genes=1, max_counts=3000, max_genes=20)
    filterCellsScanpy(data)
    X = data.get_feature(return_type="numpy")
    assert X.shape[1] == data.shape[1]


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


def test_hvg(toy_data, subtests):
    adata, data = toy_data
    with subtests.test("HighlyVariableGenesRawCount"):
        hvg = HighlyVariableGenesRawCount(n_top_genes=20)
        hvg(data)
        assert adata.X.shape[0] == data.data.X.shape[0]
    with subtests.test("HighlyVariableGenesLogarithmizedByTopGenes"):
        hvg = HighlyVariableGenesLogarithmizedByTopGenes()
        hvg(data)
        assert adata.X.shape[0] == data.data.X.shape[0]
    with subtests.test("HighlyVariableGenesLogarithmizedByMeanAndDisp"):
        hvg = HighlyVariableGenesLogarithmizedByMeanAndDisp()
        hvg(data)
        assert adata.X.shape[0] == data.data.X.shape[0]
