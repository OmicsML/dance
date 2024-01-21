import os

import numpy as np
import torch
from step3_config import get_optimizer, get_transforms

from dance import logger
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdcc import ScDCC
from dance.registry import DotDict  # 可以用可以不用
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.preprocess import generate_random_pair
from dance.utils import set_seed

fun_list = ["filter_cell_by_count", "filter_gene_by_count", "normalize_total"]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def objective(trial):
    """Optimization function."""
    parameters_dict = {
        'seed': 0,
        'num_runs': 1,
        'cache': True,
        'label_cells_files': 'label_10X_PBMC.txt',
        'label_cells': 0.1,
        'n_pairwise': 0,
        'n_pairwise_error': 0,
        'z_dim': 32,
        'encodeLayer': [256, 64],
        'sigma': 2.5,
        'gamma': 1.0,
        'lr': 0.01,
        'pretrain_lr': 0.001,
        'ml_weight': 1.0,
        'cl_weight': 1.0,
        'update_interval': 1.0,
        'tol': 0.00001,
        'ae_weights': None,
        'ae_weight_file': "AE_weights.pth.tar",
        'pretrain_epochs': 50,
        'epochs': 500,
        'batch_size': 256
    }
    transforms = get_transforms(trial=trial, fun_list=fun_list, set_data_config=False, save_raw=True)
    transforms.append(
        SetConfig({
            "feature_channel": [None, None, "n_counts"],
            "feature_channel_type": ["X", "raw_X", "obs"],
            "label_channel": "Group"
        }))
    preprocessing_pipeline = Compose(*transforms, log_level="INFO")
    parameters_config = {}
    parameters_config.update(parameters_dict)
    parameters_config = DotDict(parameters_config)
    aris = []
    for seed in range(parameters_config.seed, parameters_config.seed + parameters_config.num_runs):
        set_seed(seed)
        dataset = "10X_PBMC"
        # Load data and perform necessary preprocessing
        dataloader = ClusteringDataset("./test_automl/data", dataset=dataset)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=parameters_config.cache)

        # inputs: x, x_raw, n_counts
        inputs, y = data.get_train_data()
        n_clusters = len(np.unique(y))
        in_dim = inputs[0].shape[1]

        # Generate random pairs
        if not os.path.exists(parameters_config.label_cells_files):
            indx = np.arange(len(y))
            np.random.shuffle(indx)
            label_cell_indx = indx[0:int(np.ceil(parameters_config.label_cells * len(y)))]
        else:
            label_cell_indx = np.loadtxt(parameters_config.label_cells_files, dtype=np.int)

        if parameters_config.n_pairwise > 0:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num = generate_random_pair(y, label_cell_indx,
                                                                                 parameters_config.n_pairwise,
                                                                                 parameters_config.n_pairwise_error)
            print("Must link paris: %d" % ml_ind1.shape[0])
            print("Cannot link paris: %d" % cl_ind1.shape[0])
            print("Number of error pairs: %d" % error_num)
        else:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])

        # Build and train moodel
        model = ScDCC(input_dim=in_dim, z_dim=parameters_config.z_dim, n_clusters=n_clusters,
                      encodeLayer=parameters_config.encodeLayer, decodeLayer=parameters_config.encodeLayer[::-1],
                      sigma=parameters_config.sigma, gamma=parameters_config.gamma,
                      ml_weight=parameters_config.ml_weight, cl_weight=parameters_config.ml_weight, device=device,
                      pretrain_path=f"scdcc_{dataset}_pre.pkl")
        model.fit(inputs, y, lr=parameters_config.lr, batch_size=parameters_config.batch_size,
                  epochs=parameters_config.epochs, ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                  update_interval=parameters_config.update_interval, tol=parameters_config.tol,
                  pt_batch_size=parameters_config.batch_size, pt_lr=parameters_config.pretrain_lr,
                  pt_epochs=parameters_config.pretrain_epochs)

        # Evaluate model predictions
        score = model.score(None, y)
        print(f"{score=:.4f}")
        aris.append(score)

    print('scdcc')
    print(f'aris: {aris}')
    print(f'aris: {np.mean(aris)} +/- {np.std(aris)}')
    return ({"scores": np.mean(aris)})


if __name__ == "__main__":
    start_optimizer = get_optimizer(project="step3-cluster-scdcc-project", objective=objective)
    start_optimizer()
