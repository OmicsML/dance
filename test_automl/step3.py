import numpy as np
import optuna
import torch
from step3_config import get_optimizer, get_preprocessing_pipeline

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.utils import set_seed

fun_list = ["log1p", "filter_gene_by_count"]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def objective(trial):
    """Optimization function."""
    parameters_dict = {
        'batch_size': 128,
        "hidden_dims": [2000],
        'lambd': 0.005,
        'num_epochs': 10,
        'seed': 0,
        'num_runs': 1,
        'learning_rate': 0.0001
    }

    train_dataset = [753, 3285]
    test_dataset = [2695]
    tissue = "Brain"
    species = "mouse"
    dataloader = CellTypeAnnotationDataset(train_dataset=train_dataset, test_dataset=test_dataset, tissue=tissue,
                                           species=species, data_dir="./test_automl/data")
    preprocessing_pipeline = get_preprocessing_pipeline(trial=trial, fun_list=fun_list)
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
        scores.append(model.score(x_test, y_test))
    return {"scores": np.mean(scores)}


if __name__ == "__main__":
    start_optimizer = get_optimizer(project="step3-project", objective=objective)
    start_optimizer()
