from dance.transforms import CellTypeNums
import numpy as np
from anndata import AnnData
from dance.data import Data
import pytest 
import numpy as np
SEED = 123
def test_cell_type_nums():
    np.random.seed()
    num_cells = 100
    num_genes = 500
    gene_expression = np.random.default_rng(seed=SEED).random((num_cells, num_genes))
    cell_types = np.random.default_rng(seed=SEED).choice(['Type_A', 'Type_B', 'Type_C'], num_cells)
    adata = AnnData(X=gene_expression, obs={'cellType': cell_types})
    data = Data(adata.copy())
    data=CellTypeNums()(data)
    cell_type_nums=data.get_feature(return_type="numpy", channel="CellTypeNums", channel_type="uns")
    return cell_type_nums.shape[0]==len(np.unique(cell_types))
    
