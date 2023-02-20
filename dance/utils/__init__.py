import hashlib
import os
import random

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from dance import logger


def get_device(device: str) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def hexdigest(x: str, /) -> str:
    return hashlib.md5(x.encode()).hexdigest()


class SimpleIndexDataset(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x


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
