from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from anndata import AnnData
from torch import Tensor

FeatType = Literal["anndata", "numpy", "torch"]
ReturnedFeat = Union[np.ndarray, Tensor, AnnData]

__all__ = [
    "FeatType",
    "List",
    "Optional",
    "ReturnedFeat",
    "Sequence",
    "Tensor",
    "Tuple",
    "Union",
]
