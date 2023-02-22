from logging import Logger
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from anndata import AnnData
from torch import Tensor

LOGLEVELS: List[str] = ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR"]

CellIdxType = Union[int, str]
FeatType = Literal["anndata", "default", "numpy", "torch"]
NormMode = Literal["normalize", "standardize", "minmax", "l2"]
LogLevel = Literal[LOGLEVELS]
ReturnedFeat = Union[np.ndarray, Tensor, AnnData]

__all__ = [
    "Any",
    "Callable",
    "CellIdxType",
    "Dict",
    "FeatType",
    "Iterator",
    "LOGLEVELS",
    "List",
    "LogLevel",
    "Logger",
    "Mapping",
    "NormMode",
    "Optional",
    "ReturnedFeat",
    "Sequence",
    "Set",
    "Tensor",
    "Tuple",
    "Union",
]
