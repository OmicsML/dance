from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from step2_config import get_preprocessing_pipeline, log_in_wandb, setStep2

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@log_in_wandb(config=None)
def train(config):

    model = ACTINN(hidden_dims=config.hidden_dims, lambd=config.lambd, device=device)
    preprocessing_pipeline = get_preprocessing_pipeline(config=config)
    if preprocessing_pipeline is None:
        return {"scores": 0}
    train_dataset = [753, 3285]
    test_dataset = [2695]
    tissue = "Brain"
    species = "mouse"
    dataloader = CellTypeAnnotationDataset(train_dataset=train_dataset, test_dataset=test_dataset, tissue=tissue,
                                           species=species, data_dir="./test_automl/data")
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=False)

    # Obtain training and testing data
    x_train, y_train = data.get_train_data(return_type="torch")
    x_test, y_test = data.get_test_data(return_type="torch")
    x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()
    # Train and evaluate models for several rounds
    scores = []
    for seed in range(config.seed, config.seed + config.num_runs):
        set_seed(seed)

        model.fit(x_train, y_train, seed=seed, lr=config.learning_rate, num_epochs=config.num_epochs,
                  batch_size=config.batch_size, print_cost=False)
        scores.append(score := model.score(x_test, y_test))
    return {"scores": np.mean(scores)}


def startSweep(parameters_dict) -> Tuple[Dict[str, Any], Callable[..., Any]]:
    parameters_dict.update({
        'batch_size': {
            'value': 128
        },
        "hidden_dims": {
            'value': [2000]
        },
        'lambd': {
            'value': 0.005
        },
        'num_epochs': {
            'value': 50
        },
        'seed': {
            'value': 0
        },
        'num_runs': {
            'value': 1
        },
        'learning_rate': {
            'value': 0.0001
        }
    })
    sweep_config = {'method': 'grid'}
    sweep_config['parameters'] = parameters_dict
    metric = {'name': 'scores', 'goal': 'maximize'}

    sweep_config['metric'] = metric
    return sweep_config, train  #Return function configuration and training function


if __name__ == "__main__":
    function_list = setStep2(startSweep, original_list=["gene_filter", "gene_dim_reduction"])
    for func in function_list:
        func()