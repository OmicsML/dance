from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
from anndata import AnnData
from torch import Tensor

FeatType = Literal["anndata", "numpy", "torch"]
NormMode = Literal["normalize", "standardize", "minmax", "l2"]
LogLevel = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR"]
ReturnedFeat = Union[np.ndarray, Tensor, AnnData]

__all__ = [
    "Any",
    "Dict",
    "FeatType",
    "List",
    "LogLevel",
    "NormMode",
    "Optional",
    "ReturnedFeat",
    "Sequence",
    "Set",
    "Tensor",
    "Tuple",
    "Union",
]
