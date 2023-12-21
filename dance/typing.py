import os
from logging import Logger
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from anndata import AnnData
from omegaconf import DictConfig, DictKeyType, Node
from torch import Tensor

ConfigLike = Union[Dict[DictKeyType, Node], DictConfig]
PathLike = Union[str, bytes, os.PathLike]

CellIdxType = Union[int, str]
FeatType = Literal["anndata", "default", "numpy", "torch", "sparse"]
NormMode = Literal["normalize", "standardize", "minmax", "l2"]
GeneSummaryMode = Literal["sum", "cv", "rv", "var"]
LogLevel = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR"]
ReturnedFeat = Union[np.ndarray, Tensor, AnnData]

FileExistHandle = Literal["none", "warn", "error"]

__all__ = [
    "Any",
    "Callable",
    "CellIdxType",
    "ConfigLike",
    "Dict",
    "FeatType",
    "FileExistHandle",
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
