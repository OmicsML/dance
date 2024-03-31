import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

from dance import logger
from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.deepimpute import DeepImpute
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--sub_outputdim", type=int, default=512,
                        help="Output dimension - number of genes being imputed per AE.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer dimension - number of neurons in the dense layer.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--min_cells", type=float, default=.05,
                        help="Minimum proportion of cells expressed required for a gene to pass filtering")
    parser.add_argument("--data_dir", type=str, default='./temp_data', help='test directory')
    parser.add_argument("--dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--n_top", type=int, default=5, help="Number of predictors.")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of training set")
    parser.add_argument("--mask_rate", type=float, default=.1, help="Masking rate.")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--mask", type=bool, default=True, help="Mask data for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)

    params = parser.parse_args()
    print(vars(params))
    file_root_path = Path(params.root_path, params.dataset).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{params.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=params.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(params.seed)

        data = ImputationDataset(data_dir=params.data_dir, dataset=params.dataset,
                                 train_size=params.train_size).load_data()
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)
        X, X_raw, targets, predictors, mask = data.get_x(return_type="default")
        if not isinstance(X, np.ndarray):
            X = X.toarray()
            X_raw = X_raw.toarray()
        X = torch.tensor(X).float()
        X_raw = torch.tensor(X_raw).float()
        X_train = X * mask  # when mask is None, raise error
        model = DeepImpute(predictors, targets, params.dataset, params.sub_outputdim, params.hidden_dim, params.dropout,
                           params.seed, params.gpu)

        model.fit(X_train, X_train, mask, params.batch_size, params.lr, params.n_epochs, params.patience)
        imputed_data = model.predict(X_train, mask)
        score = model.score(X, imputed_data, mask, metric='RMSE')
        wandb.log({"RMSE": score})

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=params.sweep_id, count=params.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=params.summary_file_path, root_path=file_root_path)
    if params.tune_mode == "pipeline" or params.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{params.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(params.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path,
                       required_funs=["SaveRaw", "UpdateRaw", "GeneHoldout", "CellwiseMaskData", "SetConfig"],
                       required_indexes=[2, 6, sys.maxsize - 2, sys.maxsize - 1,
                                         sys.maxsize], metric="RMSE", ascending=True)
        if params.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce deepimpute benchmarks, please refer to command lines belows:

Mouse Brain
$ python deepimpute.py --dataset mouse_brain_data

Mouse Embryo
$ python deepimpute.py --dataset mouse_embryo_data

PBMC
$ python deepimpute.py --dataset pbmc_data

"""
