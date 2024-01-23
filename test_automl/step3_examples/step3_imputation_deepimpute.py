import torch

from dance import logger
from dance.automl_config.step3_config import get_optimizer, get_transforms
from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.deepimpute import DeepImpute
from dance.registry import DotDict
from dance.transforms.misc import Compose, SetConfig
from dance.utils import set_seed

fun_list = ["filter_gene_by_count", "filter_cell_by_count", "log1p", "gene_hold_out", "cell_wise_mask_data"]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import numpy as np


def objective(trial):
    parameters_dict = {
        'dropout': 0.1,
        'lr': 1e-5,
        'n_epochs': 5,
        'batch_size': 64,
        'sub_outputdim': 512,
        'hidden_dim': 256,
        'patience': 20,
        'min_cells': 0.05,
        "n_top": 5,
        "train_size": 0.9,
        "mask_rate": 0.1,
        "cache": False,
        "mask": True,  #Avoid duplication with hyperparameter processes
        "seed": 0,
        "num_runs": 1,
        "gpu": 3
    }
    parameters_config = {}
    parameters_config.update(parameters_dict)
    parameters_config = DotDict(parameters_config)
    rmses = []
    for seed in range(parameters_config.seed, parameters_config.seed + parameters_config.num_runs):
        set_seed(seed)
        dataset = "mouse_brain_data"
        data_dir = "./test_automl/data"
        dataloader = ImputationDataset(data_dir=data_dir, dataset=dataset, train_size=parameters_config.train_size)
        # preprocessing_pipeline = DeepImpute.preprocessing_pipeline(min_cells=parameters_config.min_cells, n_top=parameters_config.n_top,
        #                                                            sub_outputdim=parameters_config.sub_outputdim, mask=parameters_config.mask,
        #                                                            seed=seed, mask_rate=parameters_config.mask_rate)
        transforms = get_transforms(trial=trial, fun_list=fun_list, set_data_config=False, save_raw=True)
        if transforms is None:
            logger.warning("skip transforms")
            return {"scores": 0}
        transforms.append(
            SetConfig({
                "feature_channel": [None, None, "targets", "predictors", "train_mask"],
                "feature_channel_type": ["X", "raw_X", "uns", "uns", "layers"],
                "label_channel": [None, None],
                "label_channel_type": ["X", "raw_X"],
            }))
        preprocessing_pipeline = Compose(*transforms, log_level="INFO")
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=parameters_config.cache)

        if parameters_config.mask:
            X, X_raw, targets, predictors, mask = data.get_x(return_type="default")
        else:
            mask = None
            X, X_raw, targets, predictors = data.get_x(return_type="default")
        X = torch.tensor(X.toarray()).float()
        X_raw = torch.tensor(X_raw.toarray()).float()
        X_train = X * mask
        model = DeepImpute(predictors, targets, dataset, parameters_config.sub_outputdim, parameters_config.hidden_dim,
                           parameters_config.dropout, seed, parameters_config.gpu)

        model.fit(X_train, X_train, mask, parameters_config.batch_size, parameters_config.lr,
                  parameters_config.n_epochs, parameters_config.patience)
        imputed_data = model.predict(X_train, mask)
        score = model.score(X, imputed_data, mask, metric='RMSE')
        print("RMSE: %.4f" % score)
        rmses.append(score)

    print('deepimpute')
    print(f'rmses: {rmses}')
    print(f'rmses: {np.mean(rmses)} +/- {np.std(rmses)}')
    return ({"scores": np.mean(rmses)})


if __name__ == "__main__":
    start_optimizer = get_optimizer(project="step3-imputation-deepimpute-project", objective=objective, n_trials=10,
                                    direction="minimize")
    start_optimizer()
