#normalize_per_cell是一定要选的，因为需要n_counts
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
    aris = []
    for seed in range(config.seed, config.seed + config.num_runs):
        set_seed(seed)
        dataset = "10X_PBMC"
        # Load data and perform necessary preprocessing
        dataloader = ClusteringDataset("./test_automl/data", dataset=dataset)

        transforms = get_transforms(config=config, set_data_config=False, save_raw=True)
        if ("normalize" not in config.keys() or config.normalize != "normalize_total") or transforms is None:
            logger.warning("skip transforms")
            return {"scores": 0}
        transforms.append(
            SetConfig({
                "feature_channel": [None, None, "n_counts"],
                "feature_channel_type": ["X", "raw_X", "obs"],
                "label_channel": "Group"
            }))
        preprocessing_pipeline = Compose(*transforms, log_level="INFO")
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=config.cache)

        # inputs: x, x_raw, n_counts
        inputs, y = data.get_train_data()
        n_clusters = len(np.unique(y))
        in_dim = inputs[0].shape[1]

        # Generate random pairs
        if not os.path.exists(config.label_cells_files):
            indx = np.arange(len(y))
            np.random.shuffle(indx)
            label_cell_indx = indx[0:int(np.ceil(config.label_cells * len(y)))]
        else:
            label_cell_indx = np.loadtxt(config.label_cells_files, dtype=np.int)

        if config.n_pairwise > 0:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num = generate_random_pair(y, label_cell_indx, config.n_pairwise,
                                                                                 config.n_pairwise_error)
            print("Must link paris: %d" % ml_ind1.shape[0])
            print("Cannot link paris: %d" % cl_ind1.shape[0])
            print("Number of error pairs: %d" % error_num)
        else:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])

        # Build and train moodel
        model = ScDCC(input_dim=in_dim, z_dim=config.z_dim, n_clusters=n_clusters, encodeLayer=config.encodeLayer,
                      decodeLayer=config.encodeLayer[::-1], sigma=config.sigma, gamma=config.gamma,
                      ml_weight=config.ml_weight, cl_weight=config.ml_weight, device=device,
                      pretrain_path=f"scdcc_{dataset}_pre.pkl")
        model.fit(inputs, y, lr=config.lr, batch_size=config.batch_size, epochs=config.epochs, ml_ind1=ml_ind1,
                  ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2, update_interval=config.update_interval,
                  tol=config.tol, pt_batch_size=config.batch_size, pt_lr=config.pretrain_lr,
                  pt_epochs=config.pretrain_epochs)

        # Evaluate model predictions
        score = model.score(None, y)
        print(f"{score=:.4f}")
        aris.append(score)

    print('scdcc')
    print(f'aris: {aris}')
    print(f'aris: {np.mean(aris)} +/- {np.std(aris)}')
    return ({"scores": np.mean(aris)})


def startSweep(parameters_dict) -> Tuple[Dict[str, Any], Callable[..., Any]]:
    parameters_dict.update({
        'seed': {
            'value': 0
        },
        'num_runs': {
            'value': 1
        },
        'cache': {
            'value': True
        },
        'label_cells_files': {
            'value': 'label_10X_PBMC.txt'
        },
        'label_cells': {
            'value': 0.1
        },
        'n_pairwise': {
            'value': 0
        },
        'n_pairwise_error': {
            'value': 0
        },
        'z_dim': {
            'value': 32
        },
        'encodeLayer': {
            'value': [256, 64]
        },
        'sigma': {
            'value': 2.5
        },
        'gamma': {
            'value': 1.0
        },
        'lr': {
            'value': 0.01
        },
        'pretrain_lr': {
            'value': 0.001
        },
        'ml_weight': {
            'value': 1.0
        },
        'cl_weight': {
            'value': 1.0
        },
        'update_interval': {
            'value': 1.0
        },
        'tol': {
            'value': 0.00001
        },
        'ae_weights': {
            'value': None
        },
        'ae_weight_file': {
            'value': "AE_weights.pth.tar"
        },
        'pretrain_epochs': {
            'value': 50
        },
        'epochs': {
            'value': 500
        },
        'batch_size': {
            'value': 256
        }
    })

    sweep_config = {'method': 'grid'}
    sweep_config['parameters'] = parameters_dict
    metric = {'name': 'scores', 'goal': 'maximize'}

    sweep_config['metric'] = metric
    return sweep_config, train  #Return function configuration and training function


if __name__ == "__main__":
    """get_function_combinations."""
    function_list = setStep2(startSweep, original_list=["gene_filter", "cell_filter", "normalize"])
    for func in function_list:
        func()
