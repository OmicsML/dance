import numpy as np
from anndata import AnnData
from pandas import DataFrame

from dance.typing import List, Optional, Set


def cell_label_to_adata(cell_labels: List[Set[str]], idx_to_label: List[str],
                        obs: Optional[DataFrame] = None) -> AnnData:
    """Convert cell labels into AnnData of label matrix.

    Parameters
    ----------
    cell_labels
        List of set of str. Each set corresponds to the relevant cell types for that cell.
    idx_to_label
        List of cell type names, used to define the column orders in the label matrix.
    obs
        Observation matrix to use. If not set, use ordered integer as cell indices.

    Returns
    -------
    y_adata
        An AnnData object containing the label matrix, where each row represents a cell and each column represents a
        cell type. An entry is marked as ones if the cell is related to that particular cell type, otherwise it is set
        to zero.

    """
    num_samples = len(cell_labels)
    if obs is None:
        obs = DataFrame(index=list(range(num_samples)))
    elif obs.shape[0] != num_samples:
        raise IndexError(f"Mismatched sizes between cell_labels (n={num_samples:,}) and obs (n={obs.shape[0]:,})")

    num_labels = len(idx_to_label)
    var = DataFrame(index=idx_to_label)
    label_to_idx = {j: i for i, j in enumerate(idx_to_label)}

    y = np.zeros((num_samples, num_labels))
    for i in range(num_samples):
        for j in map(label_to_idx.get, cell_labels[i]):
            y[i, j] = 1

    return AnnData(X=y, obs=obs, var=var, dtype="float32")
