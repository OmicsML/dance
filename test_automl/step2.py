from itertools import combinations

from fun2code import fun2code_dict
from step2_config import getFunConfig

import wandb


def getSweepId(selected_keys=["normalize", "gene_filter", "gene_dim_reduction"]):
    pipline2fun_dict, count = getFunConfig(selected_keys)
    parameters_dict = pipline2fun_dict
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
    sweep_id = wandb.sweep(sweep_config, project="pytorch-cell_type_annotation_ACTINN")
    return sweep_id, count


import numpy as np
import torch

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.transforms.misc import Compose, SetConfig
from dance.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        if ("normalize" not in config.keys() or config.normalize
                != "log1p") and ("gene_filter" in config.keys() and config.gene_filter == "highly_variable_genes"):
            wandb.log({"scores": 0})
            return
        model = ACTINN(hidden_dims=config.hidden_dims, lambd=config.lambd, device=device)
        transforms = []
        transforms.append(fun2code_dict[config.normalize]) if "normalize" in config.keys() else None
        transforms.append(fun2code_dict[config.gene_filter]) if "gene_filter" in config.keys() else None
        transforms.append(fun2code_dict[config.gene_dim_reduction]) if "gene_dim_reduction" in config.keys() else None
        data_config = {"label_channel": "cell_type"}
        if "gene_dim_reduction" in config.keys():
            data_config.update({"feature_channel": fun2code_dict[config.gene_dim_reduction].name})
        transforms.append(SetConfig(data_config))
        preprocessing_pipeline = Compose(*transforms, log_level="INFO")
        train_dataset = [753, 3285]
        test_dataset = [2695]
        tissue = "Brain"
        species = "mouse"
        dataloader = CellTypeAnnotationDataset(train_dataset=train_dataset, test_dataset=test_dataset, tissue=tissue,
                                               species=species)
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
        wandb.log({"scores": np.mean(scores)})


if __name__ == "__main__":
    original_list = ["normalize", "gene_filter", "gene_dim_reduction"]
    all_combinations = [combo for i in range(1, len(original_list) + 1) for combo in combinations(original_list, i)]
    all_combinations.append([])
    for s_key in all_combinations:
        s_list = list(s_key)
        sweep_id, count = getSweepId(s_list)
        print(s_list, count)
        wandb.agent(sweep_id, train, count=count)
