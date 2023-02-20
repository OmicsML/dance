import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dance.data import Data
from dance.transforms.filter import FilterGenesCommon


@pytest.mark.parametrize("mode", ["batch", "split"])
def test_filter_genes_common(mode):
    x1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    x2 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    var = pd.DataFrame(index=["x", "y", "z"])
    adata1 = ad.AnnData(x1, obs=pd.DataFrame(index=["a", "b", "c"]), var=var)
    adata2 = ad.AnnData(x2, obs=pd.DataFrame(index=["d", "e"]), var=var)

    adata1.obs["batch"] = 0
    adata2.obs["batch"] = 1
    adata = ad.concat((adata1, adata2))
    data = Data(adata, train_size=adata1.shape[0])

    if mode == "batch":
        FilterGenesCommon(batch_key="batch")(data)
    elif mode == "split":
        FilterGenesCommon(split_keys=["train", "test"])(data)
    else:
        assert False, f"Unknown mode {mode!r}"

    assert data.shape[1] == 1
