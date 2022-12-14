import numpy as np
from pandas import DataFrame, Index

from dance.typing import List, Optional, Sequence, Set, Union


def cell_label_to_df(cell_labels: List[Union[str, Set[str]]], idx_to_label: List[str],
                     index: Optional[Union[Sequence[str], Index]] = None) -> DataFrame:
    """Convert cell labels into AnnData of label matrix.

    Parameters
    ----------
    cell_labels
        List of str or set of str. Each corresponds to the relevant cell type(s) for that cell.
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
    num_labels = len(idx_to_label)
    label_to_idx = {j: i for i, j in enumerate(idx_to_label)}

    y = np.zeros((num_samples, num_labels), dtype=np.float32)
    for i in range(num_samples):
        for j in map(label_to_idx.get, cell_labels[i]):
            y[i, j] = 1

    return DataFrame(y, index=index, columns=idx_to_label)
