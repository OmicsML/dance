#normalize_total是一定要选的，因为需要n_counts
import os
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from step2_config import get_transforms, log_in_wandb, setStep2

from dance import logger
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdcc import ScDCC
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.preprocess import generate_random_pair
from dance.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@log_in_wandb(config=None)
def train(config):
    pass


def startSweep(parameters_dict) -> Tuple[Dict[str, Any], Callable[..., Any]]:
    pass


if __name__ == "__main__":
    """get_function_combinations."""
    function_list = setStep2(startSweep,
                             original_list=["gene_filter", "cell_filter", "normalize", "gene_hold_out_name",
                                            "mask"], required_elements=["gene_hold_out_name"])
    for func in function_list:
        func()
