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
    parameters_dict.update({
        'dropout': {
            'value': 0.1
        },
        'lr': {
            'value': 1e-5
        },
        'n_epochs': {
            'value': 500
        },
        'batch_size': {
            'value': 64
        },
        'sub_outputdim': {
            'value': 512
        },
        'hidden_dim': {
            'value': 256
        },
        'patience': {
            'value': 20
        },
        'min_cells': {
            'value': 0.05
        },
        "n_top": {
            'value': 5
        },
        "train_size": {
            "value": 0.9
        },
        "mask_rate": {
            "value": 0.1
        },
        "cache": {
            "value": False
        },
        "mask": {
            "value": True
        },
        "seed": {
            "value": 0
        },
        "num_runs": {
            "value": 1
        }
    })
    sweep_config = {'method': 'grid'}
    sweep_config['parameters'] = parameters_dict
    metric = {'name': 'scores', 'goal': 'maximize'}

    sweep_config['metric'] = metric
    return sweep_config, train  #Return function configuration and training function


if __name__ == "__main__":
    """get_function_combinations."""
    function_list = setStep2(startSweep,
                             original_list=["gene_filter", "cell_filter", "normalize", "gene_hold_out_name",
                                            "mask"], required_elements=["gene_hold_out_name"])
    for func in function_list:
        func()
