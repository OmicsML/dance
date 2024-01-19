import numpy as np
import optuna
import torch
from optuna.integration.wandb import WeightsAndBiasesCallback

import wandb
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.transforms.misc import Compose, SetConfig
from dance.utils import set_seed
from test_automl.config import fun2code_dict

fun_list = ["log1p", "filter_gene_by_count"]
wandb_kwargs = {"project": "my-project"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


@wandbc.track_in_wandb()
def objective(trial):

    transforms = []
    for f_str in fun_list:
        fun_i = eval(f_str)
        transforms.append(fun_i(trial))
    parameters_dict = {
        'batch_size': 128,
        "hidden_dims": [2000],
        'lambd': 0.005,
        'num_epochs': 50,
        'seed': 0,
        'num_runs': 1,
        'learning_rate': 0.0001
    }
    data_config = {"label_channel": "cell_type"}
    feature_name = {"cell_svd", "cell_weighted_pca", "cell_pca"} & set(fun_list)
    if feature_name:
        data_config.update({"feature_channel": fun2code_dict[feature_name].name})
    transforms.append(SetConfig(data_config))
    preprocessing_pipeline = Compose(*transforms, log_level="INFO")
    train_dataset = [753, 3285]
    test_dataset = [2695]
    tissue = "Brain"
    species = "mouse"
    dataloader = CellTypeAnnotationDataset(train_dataset=train_dataset, test_dataset=test_dataset, tissue=tissue,
                                           species=species, data_dir="./test_automl/data")
    data = dataloader.load_data(transform=preprocessing_pipeline, cache=True)

    # Obtain training and testing data
    x_train, y_train = data.get_train_data(return_type="torch")
    x_test, y_test = data.get_test_data(return_type="torch")
    x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()
    # Train and evaluate models for several rounds
    scores = []
    parameter_config = {}
    parameter_config.update(parameters_dict)
    model = ACTINN(hidden_dims=parameter_config.get('hidden_dims'), lambd=parameter_config.get('lambd'), device=device)
    for seed in range(parameter_config.get('seed'), parameter_config.get('seed') + parameter_config.get('num_runs')):
        set_seed(seed)

        model.fit(x_train, y_train, seed=seed, lr=parameter_config.get('learning_rate'),
                  num_epochs=parameter_config.get('num_epochs'), batch_size=parameter_config.get('batch_size'),
                  print_cost=False)
        scores.append(score := model.score(x_test, y_test))
    wandb.log({"scores": np.mean(scores)})
    return np.mean(scores)


study = optuna.create_study()
study.optimize(objective, n_trials=10, callbacks=[wandbc])
