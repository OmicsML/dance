from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from step2_config import get_transforms, log_in_wandb, setStep2

from dance import logger
from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.deepimpute import DeepImpute
from dance.transforms.misc import Compose, SetConfig
from dance.utils import set_seed

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


@log_in_wandb(config=None)
def train(config):
    rmses = []
    for seed in range(config.seed, config.seed + config.num_runs):
        set_seed(seed)
        dataset = "mouse_brain_data"
        data_dir = "./test_automl/data"
        dataloader = ImputationDataset(data_dir=data_dir, dataset=dataset, train_size=config.train_size)
        # preprocessing_pipeline = DeepImpute.preprocessing_pipeline(min_cells=config.min_cells, n_top=config.n_top,
        #                                                            sub_outputdim=config.sub_outputdim, mask=config.mask,
        #                                                            seed=seed, mask_rate=config.mask_rate)
        transforms = get_transforms(config=config, set_data_config=False, save_raw=True)
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
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=config.cache)

        if config.mask:
            X, X_raw, targets, predictors, mask = data.get_x(return_type="default")
        else:
            mask = None
            X, X_raw, targets, predictors = data.get_x(return_type="default")
        X = torch.tensor(X.toarray()).float()
        X_raw = torch.tensor(X_raw.toarray()).float()
        X_train = X * mask
        model = DeepImpute(predictors, targets, dataset, config.sub_outputdim, config.hidden_dim, config.dropout, seed,
                           2)

        model.fit(X_train, X_train, mask, config.batch_size, config.lr, config.n_epochs, config.patience)
        imputed_data = model.predict(X_train, mask)
        score = model.score(X, imputed_data, mask, metric='RMSE')
        print("RMSE: %.4f" % score)
        rmses.append(score)

    print('deepimpute')
    print(f'rmses: {rmses}')
    print(f'rmses: {np.mean(rmses)} +/- {np.std(rmses)}')
    return ({"rmses": np.mean(rmses)})


def startSweep(parameters_dict) -> Tuple[Dict[str, Any], Callable[..., Any]]:
    parameters_dict.update({
        'dropout': {
            'value': 0.1
        },
        'lr': {
            'value': 1e-5
        },
        'n_epochs': {
            'value': 5
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
        "mask": {  #避免出现与超参数流程重复的情况，一般没有
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
    metric = {'name': 'rmses', 'goal': 'minimize'}

    sweep_config['metric'] = metric
    return sweep_config, train  #Return function configuration and training function


if __name__ == "__main__":
    """get_function_combinations."""
    function_list = setStep2(
        startSweep, original_list=["gene_filter", "cell_filter", "normalize", "gene_hold_out_name", "mask_name"],
        required_elements=["gene_hold_out_name", "mask_name"])
    for func in function_list:
        func()
