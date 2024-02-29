import argparse
import pprint
import sys
from pathlib import Path
from typing import get_args

import numpy as np
import wandb

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.singlecellnet import SingleCellNet
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.typing import LogLevel
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument(
        "--normalize", action="store_true", help="Whether to perform the normalization for SingleCellNet. "
        "Disabled by default since the CellTypeAnnotation data is already normalized")
    parser.add_argument("--num_rand", type=int, default=100)
    parser.add_argument("--num_top_gene_pairs", type=int, default=250)
    parser.add_argument("--num_top_genes", type=int, default=100)
    parser.add_argument("--num_trees", type=int, default=1000)
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--stratify", type=bool, default=True)
    parser.add_argument("--test_dataset", type=int, nargs="+", default=[1759],
                        help="List testing training dataset ids.")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", type=int, nargs="+", default=[1970], help="List of training dataset ids.")
    parser.add_argument("--valid_dataset", nargs="+", default=[1970], help="List of valid dataset ids.")
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--config_dir", default="", type=str)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")
    MAINDIR = Path(__file__).resolve().parent
    pipeline_planer = PipelinePlaner.from_config_file(f"{MAINDIR}/{args.config_dir}{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SingleCellNet(num_trees=args.num_trees)

        # Load data and perform necessary preprocessing
        data = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                         species=args.species, tissue=args.tissue, valid_dataset=args.valid_dataset,
                                         data_dir="./temp_data").load_data(cache=args.cache)
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # Obtain training and testing data
        x_train, y_train = data.get_train_data(return_type="numpy")
        y_train = y_train.argmax(1)  # convert one-hot representation into label index representation
        x_test, y_test = data.get_test_data(return_type="numpy")
        x_valid, y_valid = data.get_val_data(return_type="numpy")

        # XXX: last column for 'unsure' label by the model
        # TODO: add option to base model score function to account for unsure
        y_test = np.hstack([y_test, np.zeros((y_test.shape[0], 1))])

        # Train and evaluate the model
        model.fit(x_train, y_train, stratify=args.stratify, num_rand=args.num_rand, random_state=args.seed)
        score = model.score(x_valid, y_valid)
        test_score = model.score(x_test, y_test)
        wandb.log({"acc": score, "test_acc": test_score})

        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, f"{MAINDIR}/{args.summary_file_path}")
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       required_funs=["SCNFeature", "SetConfig"], required_indexes=[sys.maxsize - 1, sys.maxsize],
                       conf_load_path="../step3_default_params.yaml")
        if args.tune_mode == "pipeline_params":
            run_step3(MAINDIR, evaluate_pipeline, tune_mode="params", sweep_id=None,
                      step2_pipeline_planer=pipeline_planer)
"""To reproduce SingleCellNet benchmarks, please refer to command lines below:

Mouse Brain
$ python singlecellnet.py --species mouse --tissue Brain --train_dataset 753 --test_dataset 2695

Mouse Spleen
$ python singlecellnet.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python singlecellnet.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
