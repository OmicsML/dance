import argparse
import gc
import os
import pprint
import sys
import time
from pathlib import Path
from typing import get_args

import numpy as np
import torch
import wandb

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scgat import scGATAnnotator
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.typing import LogLevel
from dance.utils import set_seed



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, set to -1 for CPU")
    parser.add_argument("--hidden_channels", type=int, default=8)
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", default=[1970], help="List of training dataset ids.")
    parser.add_argument("--valid_dataset", nargs="+", default=None, help="List of valid dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")

    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    logger.info(f"Running scGAT with the following parameters:\n{pprint.pformat(vars(args))}")
    file_root_path = Path(
        args.root_path, "_".join([
            "-".join([str(num) for num in dataset])
            for dataset in [args.train_dataset, args.valid_dataset, args.test_dataset]
            if (dataset is not None and dataset != [])
        ])).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{Path(args.root_path).resolve()}/{args.tune_mode}_tuning_config.yaml")
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
            data = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                             train_dataset=args.train_dataset, valid_dataset=args.valid_dataset,
                                             data_dir="../temp_data").load_data()
            # Prepare preprocessing pipeline and apply it to data
            kwargs = {tune_mode: dict(wandb.config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            if run_idx == 0:
                print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
            preprocessing_pipeline(data)

            # Initialize model
            device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
            model = scGATAnnotator(
                hidden_channels=args.hidden_channels,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                device=device,
                random_seed=current_seed
            )

            # Train the model
            logger.info("Training scGAT model...")
            model.fit(data.data)

            # Evaluate the model
            logger.info("Evaluating...")

            # Get true labels for accuracy calculation
            _, y_val = data.get_val_data(return_type="torch")
            _, y_test = data.get_test_data(return_type="torch")
            if y_test.dim() > 1 and y_test.shape[1] > 1:
                y_test = y_test.argmax(1)  # convert to label index

            if y_val.dim() > 1 and y_val.shape[1] > 1:
                y_val = y_val.argmax(1)  # convert to label index

            # Make predictions
            # predict returns predictions for all cells (Array of shape [N_total])
            all_preds = model.predict(data.data)

            # Get test set mask to extract corresponding predictions
            pyg_data = data.data.uns['pyg_data']
            test_mask = pyg_data.test_mask.cpu().numpy()
            val_mask = pyg_data.val_mask.cpu().numpy()

            # Extract test set predictions
            y_pred = all_preds[test_mask]
            y_pred_val = all_preds[val_mask]

            # Calculate accuracy
            # Ensure y_pred and y_test have consistent lengths
            if len(y_pred) != len(y_test):
                logger.warning(f"Shape mismatch: Preds {len(y_pred)} vs Labels {len(y_test)}. "
                               "Using intersection or checking split logic.")

            run_train_score = (y_pred_val == y_val.cpu().numpy()).mean()  # using val as train score for consistency
            run_valid_score = (y_pred_val == y_val.cpu().numpy()).mean()
            run_test_score = (y_pred == y_test.cpu().numpy()).mean()

            train_scores.append(run_train_score)
            valid_scores.append(run_valid_score)
            test_scores.append(run_test_score)

            logger.info(f"Run {run_idx + 1} finished. Valid Acc: {run_valid_score:.4f}, Test Acc: {run_test_score:.4f}")

            del model, data
            gc.collect()
            if device != "cpu": torch.cuda.empty_cache()

        # Stop Timer
        total_time_seconds = time.time() - start_time

        avg_train_score = np.mean(train_scores)
        avg_valid_score = np.mean(valid_scores)
        avg_test_score = np.mean(test_scores)

        # Calculate Speed Score and Combined Score
        speed_score = 1.0 / (1.0 + total_time_seconds / 300.0)
        combined_score = 0.8 * avg_valid_score + 0.2 * speed_score

        logger.info(f"Averaged over {args.num_runs} runs - Valid Acc: {avg_valid_score:.4f}, Time: {total_time_seconds:.2f}s, Combined Score: {combined_score:.4f}")

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

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  # Score can be recorded for each epoch
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
"""To reproduce the benchmarking results, please run the following command:

Mouse Spleen
$ python main.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Brain
$ python main.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Kidney
$ python main.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""