from __future__ import annotations

import os
from logging import Logger
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
from anndata import AnnData
from omegaconf import DictConfig, DictKeyType, ListConfig, Node
from torch import Tensor

if TYPE_CHECKING:  # https://peps.python.org/pep-0563/#forward-references
    from dance.config import Config

ConfigLike = Union[Dict[DictKeyType, Node], DictConfig, ListConfig, "Config"]
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
    "Iterable",
    "Iterator",
    "List",
    "LogLevel",
    "Logger",
    "Mapping",
    "NormMode",
    "Number",
    "Optional",
    "ReturnedFeat",
    "Sequence",
    "Set",
    "Tensor",
    "Tuple",
    "Type",
    "Union",
]
