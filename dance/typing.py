from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
from anndata import AnnData
from torch import Tensor

FeatType = Literal["anndata", "numpy", "torch"]
ReturnedFeat = Union[np.ndarray, Tensor, AnnData]

__all__ = [
    "Dict",
    "FeatType",
    "List",
    "Optional",
    "ReturnedFeat",
    "Sequence",
    "Set",
    "Tensor",
    "Tuple",
    "Union",
]
