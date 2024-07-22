import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch

import wandb
from dance import logger
from dance.datasets.multimodality import ModalityPredictionDataset
from dance.modules.multi_modality.predict_modality.babel import BabelWrapper
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    OPTIMIZER_DICT = {
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2_rna")
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    parser.add_argument("-seed", "--seed", default=1, type=int)
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("-m", "--model_folder", default="./models")
    parser.add_argument("--outdir", "-o", default="./logs", help="Directory to output to")
    parser.add_argument("--lossweight", type=float, default=1., help="Relative loss weight")
    parser.add_argument("--lr", "-l", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batchsize", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimensions")
    parser.add_argument("--earlystop", type=int, default=20, help="Early stopping after N epochs")
    parser.add_argument("--naive", "-n", action="store_true", help="Use a naive model instead of lego model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=500)

    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    args = parser.parse_args()
    args.resume = True

    torch.set_num_threads(args.cpus)
    args.outdir = os.path.abspath(args.outdir)
    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    # Specify output log file
    fh = logging.FileHandler(f"{args.outdir}/training_{args.subtask}_{args.seed}.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    file_root_path = Path(args.root_path, args.subtask).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    for arg in vars(args):
        logger.info(f"Parameter {arg}: {getattr(args, arg)}")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        rndseed = args.seed
        set_seed(rndseed)
        dataset = ModalityPredictionDataset(args.subtask, preprocess=None)
        data = dataset.load_data()
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # Obtain training and testing data
        x_train, y_train = data.get_train_data(return_type="torch")
        x_test, y_test = data.get_test_data(return_type="torch")
        x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()
        # Train and evaluate the model
        #突然想到，或许有些算法可以降维，而有些算法不能降维，所以还是要依据算法而定
        model = BabelWrapper(args, dim_in=x_train.shape[1], dim_out=y_train.shape[1])
        model.fit(x_train, y_train, val_ratio=0.15)
        wandb.log({'rmse': model.score(x_test, y_test)})
        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path,
                       required_funs=["AlignMod", "FilterCellsCommonMod", "FilterCellsCommonMod",
                                      "SetConfig"], required_indexes=[2, 11, 14, sys.maxsize], metric="ARI")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce BABEL on other samples, please refer to command lines belows:

GEX to ADT (subset):
$ python babel.py --subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda

GEX to ADT:
$ python babel.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

ADT to GEX:
$ python babel.py --subtask openproblems_bmmc_cite_phase2_mod2 --device cuda

GEX to ATAC:
$ python babel.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

ATAC to GEX:
$ python babel.py --subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda

"""
