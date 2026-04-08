import argparse
import gc
import os
import pprint
import sys
import time  # <--- Added import time
from pathlib import Path
from typing import get_args

import numpy as np
import torch
import wandb

from dance import logger
from dance.data import Data
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.graphcs import GraphCSClassifier, load_GBP_data
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.transforms.filter import HighlyVariableGenesLogarithmizedByTopGenes, SupervisedFeatureSelection
from dance.transforms.graph.graphcs import BBKNNConstruction
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.normalize import NormalizeTotalLog1P
from dance.typing import LogLevel
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base Dance arguments
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dense_dim", type=int, default=400, help="dim of PCA")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, set to -1 for CPU")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], type=int, help="list of dataset id")
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], type=int, help="list of dataset id")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")

    # Training parameters (passed to GraphCSClassifier.__init__)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--vat_lr", type=float, default=0.1, help="VAT learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")

    # Model architecture parameters
    parser.add_argument("--layer", type=int, default=2, help="number of layers")
    parser.add_argument("--hidden", type=int, default=256, help="hidden dimensions")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--bias", default='none', help="bias usage")

    # GraphCS specific parameters (if used in preprocessing or internal logic)
    parser.add_argument("--alpha", type=float, default=0.05, help="decay factor (GraphCS)")
    parser.add_argument("--rmax", type=float, default=1e-5, help="threshold (GraphCS)")
    parser.add_argument("--rrz", type=float, default=0.5, help="gamma/rrz (GraphCS)")

    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--filetype", default="csv")
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    args = parser.parse_args()

    # Update GPU argument for the model wrapper expectations
    # The class expects args.gpus to be a list
    args.gpus = [args.gpu] if args.gpu != -1 else []

    logger.setLevel(args.log_level)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    logger.info(f"Running GraphCS with the following parameters:\n{pprint.pformat(vars(args))}")

    # Construct file root path similar to scdeepsort
    file_root_path = Path(
        args.root_path, "_".join([
            "-".join([str(num) for num in dataset]) for dataset in [args.train_dataset, args.test_dataset]
            if (dataset is not None and dataset != [])
        ])).resolve()
    logger.info(f"\n files is saved in {file_root_path}")

    pipeline_planer = PipelinePlaner.from_config_file(
        f"{Path(args.root_path).resolve()}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    # ================= MODIFIED FUNCTION STARTS HERE =================
    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))

        train_scores = []
        valid_scores = []
        test_scores = []

        # Start Timer
        start_time = time.time()

        for run_idx in range(args.num_runs):
            logger.info(f"Starting Run {run_idx + 1}/{args.num_runs}")

            current_seed = args.seed + run_idx
            set_seed(current_seed)

            # Load data and perform necessary preprocessing
            data = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                             species=args.species, tissue=args.tissue,
                                             val_size=args.val_size).load_data()

            # Prepare preprocessing pipeline and apply it to data
            kwargs = {tune_mode: dict(wandb.config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            if run_idx == 0:
                print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
            preprocessing_pipeline(data)

            # Initialize model: args are passed here, so self.batch_size etc are set now
            model = GraphCSClassifier(args, random_state=current_seed)

            # Get preprocessing pipeline from model (this might be different from pipeline_planer)

            X, y = data.get_data(return_type="torch")
            temp_graph = data.data.uns["temp_graph"]
            dataset_name = f"{args.species}_{args.tissue}"
            features = load_GBP_data(dataset_name, args.alpha, args.rmax, args.rrz, X, temp_graph)

            # Obtain training and testing data
            x_train = features[data.train_idx]
            x_val = features[data.val_idx]
            x_test = features[data.test_idx]
            y_train = y[data.train_idx]
            y_val = y[data.val_idx]
            y_test = y[data.test_idx]

            # Convert OneHot labels to Integer labels for PyTorch CrossEntropy
            if y_train.shape[1] > 1:
                y_train_converted = y_train.argmax(1)
            else:
                y_train_converted = y_train.flatten()

            if y_val.shape[1] > 1:
                y_val_converted = y_val.argmax(1)
            else:
                y_val_converted = y_val.flatten()

            nfeat = x_train.shape[1]
            nclass = max(int(y_train_converted.max()), int(y_val_converted.max())) + 1

            # Train: fit() uses self.batch_size initialized earlier
            model.fit(torch.FloatTensor(x_train), torch.LongTensor(y_train_converted), torch.FloatTensor(x_val),
                      torch.LongTensor(y_val_converted), nfeat, nclass)

            # Predict/Score: uses self.batch_size
            run_train_score = model.score(x_train, y_train)
            run_valid_score = model.score(x_val, y_val)
            run_test_score = model.score(x_test, y_test)

            train_scores.append(run_train_score)
            valid_scores.append(run_valid_score)
            test_scores.append(run_test_score)

            logger.info(f"Run {run_idx + 1} finished. Valid Acc: {run_valid_score:.4f}, Test Acc: {run_test_score:.4f}")

            del model, data
            gc.collect()
            if args.gpu != -1:  # Use args.gpu instead of device
                torch.cuda.empty_cache()

        # Stop Timer
        total_time_seconds = time.time() - start_time

        avg_train_score = np.mean(train_scores)
        avg_valid_score = np.mean(valid_scores)
        avg_test_score = np.mean(test_scores)

        # Calculate Speed Score and Combined Score
        speed_score = 1.0 / (1.0 + total_time_seconds / 300.0)
        combined_score = 0.8 * avg_valid_score + 0.2 * speed_score

        logger.info(
            f"Averaged over {args.num_runs} runs - Valid Acc: {avg_valid_score:.4f}, Time: {total_time_seconds:.2f}s, Combined Score: {combined_score:.4f}"
        )

        wandb.log({
            "train_acc": avg_train_score,
            "acc": avg_valid_score,
            "test_acc": avg_test_score,
            "time": total_time_seconds,
            "speed_score": speed_score,
            "combined_score": combined_score
        })
        wandb.finish()

    # ================= MODIFIED FUNCTION ENDS HERE =================

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(evaluate_pipeline, sweep_id=args.sweep_id,
                                                                  count=args.count)
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path,
                      additional_sweep_ids=args.additional_sweep_ids)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(
            result_load_path=f"{args.summary_file_path}",
            step2_pipeline_planer=pipeline_planer,
            conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
            root_path=file_root_path,
            required_funs=["CellFeatureGraph", "SetConfig"],
            required_indexes=[sys.maxsize - 1, sys.maxsize],
        )
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce GraphCS benchmarks, please refer to command lines below:

Mouse Brain
$ python graphcs.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695 --lr 0.001 --hidden 256

Mouse Spleen
$ python graphcs.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --vat_lr 0.1

Mouse Kidney
$ python graphcs.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

$ python graphcs.py --species human --tissue Brain --train_dataset 328 --test_dataset 138
"""
