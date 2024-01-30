import numpy as np
import pandas as pd
from anndata import AnnData

from dance.data import Data
from dance.transforms.graph import RESEPTGraph

SEED = 123


def test_RESPET_GRAPH():
    num_cells = 100
    num_genes = 500
    gene_expression = np.random.default_rng(seed=SEED).random((num_cells, num_genes))
    adata = AnnData(X=gene_expression)
    random_df = pd.DataFrame(
        np.random.default_rng(seed=SEED).integers(1, 10000, size=(num_cells, 2)), columns=["x_pixel", "y_pixel"],
        index=adata.obs_names)
    adata.obsm['spatial_pixel'] = random_df
    data = Data(adata.copy())
    RESEPTgraph = RESEPTGraph()
    RESEPTgraph(data)
    assert data.data.uns['RESEPTGraph'].shape == (2000, 2000, 3)
