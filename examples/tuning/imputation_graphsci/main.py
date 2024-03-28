import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

from dance import logger
from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.graphsci import GraphSCI
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of training set")
    parser.add_argument("--le", type=float, default=1, help="parameter of expression loss")
    parser.add_argument("--la", type=float, default=1e-9, help="parameter of adjacency loss")
    parser.add_argument("--ke", type=float, default=1e2, help="parameter of KL divergence of expression")
    parser.add_argument("--ka", type=float, default=1, help="parameter of KL divergence of adjacency")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--data_dir", type=str, default='./temp_data', help='test directory')
    parser.add_argument("--save_dir", type=str, default='result', help='save directory')
    parser.add_argument("--filetype", type=str, default='h5', choices=['csv', 'gz', 'h5'],
                        help='data file type, csv, csv.gz, or h5')
    parser.add_argument("--dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for exponential LR decay.")
    parser.add_argument("--threshold", type=float, default=.3,
                        help="Lower bound for correlation between genes to determine edges in graph.")
    parser.add_argument("--mask_rate", type=float, default=.1, help="Masking rate.")
    parser.add_argument("--min_cells", type=float, default=.05,
                        help="Minimum proportion of cells expressed required for a gene to pass filtering")
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
        gpu = params.gpu
        device = "cpu" if gpu == -1 else f"cuda:{gpu}"

        data = ImputationDataset(data_dir=params.data_dir, dataset=params.dataset,
                                 train_size=params.train_size).load_data()
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        X, X_raw, g, mask = data.get_x(return_type="default")
        if not isinstance(X, np.ndarray):
            X = X.toarray()
            X_raw = X_raw.toarray()
        X = torch.tensor(X).float()
        X_raw = torch.tensor(X_raw).float()
        X_train = (X * mask).to(device)
        X_raw_train = (X_raw * mask).to(device)
        g = g.to(device)

        model = GraphSCI(num_cells=X.shape[0], num_genes=X.shape[1], dataset=params.dataset, dropout=params.dropout,
                         gpu=gpu, seed=params.seed)
        model.fit(X_train, X_raw_train, g, mask, params.le, params.la, params.ke, params.ka, params.n_epochs, params.lr,
                  params.weight_decay)
        model.load_model()
        imputed_data = model.predict(X_train, X_raw_train, g, mask)
        score = model.score(X, imputed_data, mask, metric='RMSE')
        wandb.log({"RMSE": score})
        gc.collect()
        torch.cuda.empty_cache()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=params.sweep_id, count=params.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=params.summary_file_path, root_path=file_root_path)
    if params.tune_mode == "pipeline" or params.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{params.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(params.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path,
                       required_funs=["SaveRaw", "FeatureFeatureGraph", "CellwiseMaskData",
                                      "SetConfig"], required_indexes=[2, sys.maxsize - 2, sys.maxsize - 1,
                                                                      sys.maxsize], metric="RMSE", ascending=True)
        if params.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce GraphSCI benchmarks, please refer to command lines belows:

Mouse Brain:
CUDA_VISIBLE_DEVICES=2 python graphsci.py --dataset mouse_brain_data

Mouse Embryo:
CUDA_VISIBLE_DEVICES=2 python graphsci.py --dataset mouse_embryo_data

PBMC
CUDA_VISIBLE_DEVICES=2 python graphsci.py --dataset pbmc_data

"""
