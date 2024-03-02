import argparse
import pprint
import sys
from pathlib import Path
from typing import get_args

import numpy as np
import wandb

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.typing import LogLevel
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--device", default="cpu", help="Computation device.")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[2000], help="Hidden dimensions.")
    parser.add_argument("--lambd", type=float, default=0.005, help="Regularization parameter")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--nofilter", action="store_true", help="Disable filtering genes by expression summaries.")
    parser.add_argument(
        "--normalize", action="store_true", help="Whether to perform the normalization described in ACTINN. "
        "Disabled by default since the CellTypeAnnotation data is already normalized")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--print_cost", action="store_true", help="Print cost when training")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], help="List of testing dataset ids.")
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], help="List of training dataset ids.")
    parser.add_argument("--valid_dataset", nargs="+", default=[1970], help="List of valid dataset ids.")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")

    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"\n{pprint.pformat(vars(args))}")
    file_root_path = Path(
        args.root_path, "_".join([
            "/".join([str(num) for num in dataset])
            for dataset in [args.train_dataset, args.valid_dataset, args.test_dataset]
        ])).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Initialize model and get model specific preprocessing pipeline
        model = ACTINN(hidden_dims=args.hidden_dims, lambd=args.lambd, device=args.device, random_seed=args.seed)
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)

        # Load data and perform necessary preprocessing
        data = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                         valid_dataset=args.valid_dataset, data_dir="./temp_data", tissue=args.tissue,
                                         species=args.species).load_data()

        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # Obtain training and testing data
        x_train, y_train = data.get_train_data(return_type="torch")
        x_test, y_test = data.get_test_data(return_type="torch")
        x_valid, y_valid = data.get_val_data(return_type="torch")

        # Train and evaluate models for several rounds

        model.fit(x_train, y_train, seed=args.seed, lr=args.learning_rate, num_epochs=args.num_epochs,
                  batch_size=args.batch_size, print_cost=args.print_cost)
        score = model.score(x_valid, y_valid)
        test_score = model.score(x_test, y_test)
        wandb.log({"acc": score, "test_acc": test_score})
        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path)
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce ACTINN benchmarks, please refer to command lines below:

Mouse Brain
$ python actinn.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python actinn.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python actinn.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
