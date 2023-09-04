import os
from logging import Logger
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from anndata import AnnData
from torch import Tensor

PathLike = Union[str, bytes, os.PathLike]

CellIdxType = Union[int, str]
FeatType = Literal["anndata", "default", "numpy", "torch", "sparse"]
NormMode = Literal["normalize", "standardize", "minmax", "l2"]
GeneSummaryMode = Literal["sum", "cv", "rv", "var"]
LogLevel = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR"]
ReturnedFeat = Union[np.ndarray, Tensor, AnnData]

__all__ = [
    "Any",
    "Callable",
    "CellIdxType",
    "Dict",
    "FeatType",
    "GeneSummaryMode",
    "Iterator",
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
