import argparse
import gc
import sys
import pandas as pd

from sklearn.utils import issparse
import torch
from pathlib import Path
import numpy as np
import wandb
from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.magic import MAGIC
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed
import os
from dance import logger
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=6)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--ka", type=int, default=10)
    parser.add_argument("--epsilon", type=int, default=1)
    parser.add_argument("--rescale", type=int, default=0)#99
    parser.add_argument("--dim",type=int,default=20)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 for cpu")
    parser.add_argument("--min_cells", type=float, default=.01,
                        help="Minimum proportion of cells expressed required for a gene to pass filtering")
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument("--dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--mask", type=bool, default=True, help="Mask data for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of training set")
    parser.add_argument("--mask_rate", type=float, default=.1, help="Masking rate.")
    
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--get_result", action="store_true",help="save imputation result")
    params = parser.parse_args()
    print(vars(params))
    file_root_path = Path(params.root_path, params.dataset).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{params.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    
    
    def evaluate_pipeline(tune_mode=params.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(params.seed)
        seed=params.seed
        set_seed(seed)
        data = ImputationDataset(data_dir=params.data_dir, dataset=params.dataset, train_size=params.train_size).load_data()
        #change
        wandb_config = wandb.config
        if "run_kwargs" in pipeline_planer.config:
            if any(d == dict(wandb.config["run_kwargs"]) for d in pipeline_planer.config.run_kwargs):
                wandb_config = wandb_config["run_kwargs"]
            else:
                wandb.log({"skip": 1})
                wandb.finish()
                return
        kwargs = {tune_mode: dict(wandb_config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        
        preprocessing_pipeline(data)

        if params.mask:
            X, X_raw, mask,X_pca = data.get_x(return_type="default")
        else:
            mask = None
            X, X_raw, X_pca = data.get_x(return_type="default")
        X = torch.tensor(X.toarray() if issparse(X) else np.array(X)).float()
        X_raw = torch.tensor(X_raw.toarray() if issparse(X_raw) else np.array(X)).float()
        X_train = X * mask
        model = MAGIC(t=params.t, k=params.k, ka=params.ka, epsilon=params.epsilon, rescale=params.rescale,gpu=params.gpu)

        # model.fit(X_train, X_train, mask, params.batch_size, params.lr, params.n_epochs, params.patience)
        imputed_data = model.predict(X_train,X_pca)
        score = model.score(X, imputed_data, mask, metric='RMSE')
        pcc = model.score(X, imputed_data, mask, metric='PCC')
        mre = model.score(X, imputed_data, mask, metric='MRE')
        wandb.log({"RMSE": score, "PCC": pcc, "MRE": mre})
        gc.collect()
        torch.cuda.empty_cache()
        if params.get_result:
            result=model.predict(X,X_pca)
            array = result.detach().cpu().numpy()
            # Create DataFrame
            df = pd.DataFrame(
                data=array,
                index=data.data.obs_names,
                columns=data.data.var_names
            )
            df.to_csv(f"{params.dataset}/result.csv")
    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=params.sweep_id, count=params.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=params.summary_file_path, root_path=file_root_path)
    if params.get_result:
        sys.exit(0)
    if params.tune_mode == "pipeline" or params.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{params.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(params.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path,
                       required_funs=["SaveRaw", "UpdateRaw", "FeatureFeatureGraph", "CellwiseMaskData", "SetConfig"],
                       required_indexes=[2, 6, sys.maxsize - 2, sys.maxsize - 1,
                                         sys.maxsize], metric="MRE", ascending=True)
        if params.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
   
"""To reproduce deepimpute benchmarks, please refer to command lines belows:

Mouse Brain
$ python magic.py --dataset mouse_brain_data

Mouse Embryo
$ python magic.py --dataset mouse_embryo_data

PBMC
$ python magic.py --dataset pbmc_data

"""
