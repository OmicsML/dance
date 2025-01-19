# test_anndata_similarity.py

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from dance.atlas.sc_similarity.anndata_similarity import AnnDataSimilarity


@pytest.fixture
def test_data():
    """创建测试用的AnnData对象."""
    n_cells = 20
    n_genes = 40

    # 使用稀疏的非负数据来模拟单细胞数据
    X1 = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    X2 = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))

    # 确保数据是稀疏的并转换为CSR格式
    X1[np.random.random(X1.shape) > 0.5] = 0
    X2[np.random.random(X2.shape) > 0.5] = 0
    X1 = sparse.csr_matrix(X1)
    X2 = sparse.csr_matrix(X2)

    obs1 = {
        'celltype': ['type1'] * n_cells,
        'nnz': [X1[i].count_nonzero() for i in range(n_cells)],
        'n_measured_vars': [n_genes] * n_cells,
        'assay': ['assay1'] * n_cells,
        'tissue': ['tissue1'] * n_cells
    }

    obs2 = {
        'celltype': ['type2'] * n_cells,
        'nnz': [X2[i].count_nonzero() for i in range(n_cells)],
        'n_measured_vars': [n_genes] * n_cells,
        'assay': ['assay2'] * n_cells,
        'tissue': ['tissue2'] * n_cells
    }

    adata1 = AnnData(X1, obs=obs1)
    adata2 = AnnData(X2, obs=obs2)

    # 进行标准的单细胞数据预处理
    for adata in [adata1, adata2]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata1, adata2


def test_similarity_computation(test_data):
    """测试相似性计算的正确性."""
    adata1, adata2 = test_data

    # 初始化相似性计算器
    similarity_calculator = AnnDataSimilarity(adata1, adata2, sample_size=2, init_random_state=42, n_runs=1)

    # 计算相似性
    similarity_matrices = similarity_calculator.get_similarity_matrix_A2B(methods=[
        'wasserstein', 'Hausdorff', 'chamfer', 'energy', 'sinkhorn2', 'bures', 'spectral', 'common_genes_num',
        'metadata_sim', 'mmd'
    ])
    for k, v in similarity_matrices.items():
        print(k, v)

    # 验证结果格式和基本属性
    assert isinstance(similarity_matrices, dict)
    assert all(0 <= v <= 1 for k, v in similarity_matrices.items() if k not in ['common_genes_num', 'bures'])

    # 验证特定指标
    assert similarity_matrices['common_genes_num'] >= 0  # 至少有0个共同基因
    assert np.isclose(similarity_matrices['metadata_sim'], 0.0)  # 不同的元数据


def test_preprocess(test_data):
    """测试数据预处理功能."""
    adata1, adata2 = test_data

    calculator = AnnDataSimilarity(adata1, adata2)
    calculator.preprocess()

    # 验证基因过滤
    assert len(calculator.common_genes) > 0
    assert all(gene in calculator.origin_adata1.var_names for gene in calculator.common_genes)
    assert all(gene in calculator.origin_adata2.var_names for gene in calculator.common_genes)


def test_sample_cells(test_data):
    """测试细胞采样功能."""
    adata1, adata2 = test_data
    sample_size = 1

    calculator = AnnDataSimilarity(adata1, adata2, sample_size=sample_size)
    calculator.adata1 = calculator.origin_adata1.copy()
    calculator.adata2 = calculator.origin_adata2.copy()
    calculator.sample_cells(random_state=42)

    # 验证采样大小
    assert calculator.sampled_adata1.n_obs == sample_size
    assert calculator.sampled_adata2.n_obs == sample_size


@pytest.mark.xfail(raises=ValueError, strict=True)
def test_invalid_method(test_data):
    """测试无效的相似性计算方法."""
    adata1, adata2 = test_data
    calculator = AnnDataSimilarity(adata1, adata2)

    calculator.compute_similarity(random_state=42, methods=['invalid_method'])
