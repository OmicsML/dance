# test_anndata_similarity.py

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from dance.atlas.sc_similarity.anndata_similarity import AnnDataSimilarity


@pytest.fixture
def test_data():
    """Create test AnnData objects."""
    n_cells = 20
    n_genes = 40

    X1 = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    X2 = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))

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

    # Perform standard single-cell data preprocessing
    for adata in [adata1, adata2]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata1, adata2


def test_similarity_computation(test_data):
    """Test the correctness of similarity computation."""
    adata1, adata2 = test_data

    # Initialize similarity calculator
    similarity_calculator = AnnDataSimilarity(adata1, adata2, sample_size=2, init_random_state=42, n_runs=1)

    # Calculate similarity
    similarity_matrices = similarity_calculator.get_similarity_matrix_A2B(methods=[
        'wasserstein', 'Hausdorff', 'chamfer', 'energy', 'sinkhorn2', 'bures', 'spectral', 'common_genes_num',
        'metadata_sim', 'mmd'
    ])
    for k, v in similarity_matrices.items():
        print(k, v)

    # Verify result format and basic properties
    assert isinstance(similarity_matrices, dict)
    assert all(0 <= v <= 1 for k, v in similarity_matrices.items() if k not in ['common_genes_num', 'bures'])

    # Verify specific metrics
    assert similarity_matrices['common_genes_num'] >= 0  # At least 0 common genes


def test_preprocess(test_data):
    """Test data preprocessing functionality."""
    adata1, adata2 = test_data

    calculator = AnnDataSimilarity(adata1, adata2)
    calculator.preprocess()

    # Verify gene filtering
    assert len(calculator.common_genes) > 0
    assert all(gene in calculator.origin_adata1.var_names for gene in calculator.common_genes)
    assert all(gene in calculator.origin_adata2.var_names for gene in calculator.common_genes)


def test_sample_cells(test_data):
    """Test cell sampling functionality."""
    adata1, adata2 = test_data
    sample_size = 1

    calculator = AnnDataSimilarity(adata1, adata2, sample_size=sample_size)
    calculator.adata1 = calculator.origin_adata1.copy()
    calculator.adata2 = calculator.origin_adata2.copy()
    calculator.sample_cells(random_state=42)

    # Verify sample size
    assert calculator.sampled_adata1.n_obs == sample_size
    assert calculator.sampled_adata2.n_obs == sample_size


@pytest.mark.xfail(raises=ValueError, strict=True)
def test_invalid_method(test_data):
    """Test invalid similarity computation method."""
    adata1, adata2 = test_data
    calculator = AnnDataSimilarity(adata1, adata2)

    calculator.compute_similarity(random_state=42, methods=['invalid_method'])
