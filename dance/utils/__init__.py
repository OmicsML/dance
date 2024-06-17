# TODO: move these to utils.misc, __init__ should only be used for importing
import hashlib
import importlib
import os
import random
import warnings
from typing import get_args

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from dance import logger
from dance.typing import Any, FileExistHandle, Optional, PathLike


def get_device(device: str) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def hexdigest(x: str, /) -> str:
    return hashlib.md5(x.encode()).hexdigest()


def default(value: Any, default_value: Any):
    return default_value if value is None else value


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class SimpleIndexDataset(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x


class Color:
    COLOR_DICT = {
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
    }
    ENDC = "\033[0m"

    def __init__(self, color: str):
        if (code := self.COLOR_DICT.get(color)) is None:
            raise ValueError(f"Unknown color {color}, supported options: {sorted(self.COLOR_DICT)}")
        self._start = code

    @property
    def start(self) -> str:
        return self._start

    @property
    def end(self) -> str:
        return self.ENDC

    def __call__(self, txt: str) -> str:
        return "".join((self.start, txt, self.end))


def set_seed(rndseed, cuda: bool = True, extreme_mode: bool = False):
    os.environ["PYTHONHASHSEED"] = str(rndseed)
    random.seed(rndseed)
    np.random.seed(rndseed)
    torch.manual_seed(rndseed)
    if cuda:
        torch.cuda.manual_seed(rndseed)
        torch.cuda.manual_seed_all(rndseed)
    if extreme_mode:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dgl.seed(rndseed)
    dgl.random.seed(rndseed)
    logger.info(f"Setting global random seed to {rndseed}")


def file_check(path: PathLike, exist_handle: FileExistHandle = "none"):
    """Check if file exists and handle accordingly."""
    if not os.path.isfile(path):
        return

    if exist_handle == "warn":
        warnings.warn(f"File exists! {path}", UserWarning, stacklevel=3)
    elif exist_handle == "error":
        raise FileExistsError(path)
    elif exist_handle != "none":
        raise ValueError(f"Unknwon file exist handling: {exist_handle!r}, "
                         f"supported options are: {get_args(FileExistHandle)}")


def try_import(module_name: str, install_name: Optional[str] = None):
    install_name = default(install_name, module_name)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(f"{module_name} not installed. Please install first: $ pip install {install_name}") from e
