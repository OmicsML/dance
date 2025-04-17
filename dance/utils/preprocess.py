import numpy as np
from pandas import DataFrame, Index

from dance import logger
from dance.typing import List, Optional, Sequence, Set, Union


def cell_label_to_df(cell_labels: List[Optional[Union[str, Set[str]]]], idx_to_label: Optional[List[str]] = None,
                     index: Optional[Union[Sequence[str], Index]] = None) -> DataFrame:
    """Convert cell labels into AnnData of label matrix.

    Parameters
    ----------
    cell_labels
        List of str or set of str (or ``None`` if the cell type information is not available or do not match the
        training cell type data). Each corresponds to the relevant cell type(s) for that cell.
    idx_to_label
        List of cell type names, used to define the column orders in the label matrix. If not set, then use the sorted
        unique items.
    index
        Index to use for setting the observation matrix. If not set, then use range.

    Returns
    -------
    ct_label_df
        An pandas DataFrame containing the label matrix, where each row represents a cell and each column represents a
        cell type. An entry is marked as ones if the cell is related to that particular cell type, otherwise it is set
        to zero.

    """
    if idx_to_label is None:
        idx_to_label = sorted(set(cell_labels))

    num_samples = len(cell_labels)
    num_labels = len(idx_to_label)
    label_to_idx = {j: i for i, j in enumerate(idx_to_label)}

    miss_counts = 0
    y = np.zeros((num_samples, num_labels), dtype=np.float32)
    for i, labels in enumerate(cell_labels):
        if labels is None:
            miss_counts += 1
            logger.debug(f"Cell #{i} did not match any cell type in the training set.")
            continue
        elif isinstance(labels, str):
            labels = [labels]

        for j in map(label_to_idx.get, labels):
            y[i, j] = 1

    ct_label_df = DataFrame(y, index=index, columns=idx_to_label)

    if miss_counts > 0:
        logger.warning(f"{miss_counts:,} cells (out of {num_samples:,}) did not match any training cell-types.")

    return ct_label_df
