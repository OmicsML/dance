import argparse
import gc
import os
import pprint
import sys
import time  # <--- Added import time
from pathlib import Path
from typing import get_args

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scheteronet import (
    convert_dgl_to_original_format,
    eval_acc,
    print_statistics,
    scHeteroNet,
    set_split,
)
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.transforms.misc import Compose, SaveRaw
from dance.typing import LogLevel
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... [Arguments remain unchanged] ...
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--valid_dataset", nargs="+", default=None, help="List of valid dataset ids.")
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument('--data_dir', type=str, default='../temp_data')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_step', type=int, default=1, help='how often to print')
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument('--train_prop', type=float, default=.6, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2, help='validation label proportion')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'], help='evaluation metric')
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers for deep methods')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of mlp layers')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--oodprop', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--oodalpha', type=float, default=0.3, help='weight for residual connection in propagation')
    parser.add_argument('--use_zinb', action='store_true', help='whether to use ZINB loss')
    parser.add_argument('--use_2hop', action='store_false', help='whether to use 2-hop propagation')
    parser.add_argument('--zinb_weight', type=float, default=1e-4)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument('--display_step', type=int, default=10, help='how often to print')
    parser.add_argument('--print_prop', action='store_true', help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true', help='print args for hyper-parameter searching')
    parser.add_argument('--cl_weight', type=float, default=0.0)
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument('--spatial', action='store_false', help='read spatial')
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--filetype", default="csv")
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    logger.info(f"Running ScDeepSort with the following parameters:\n{pprint.pformat(vars(args))}")

    file_root_path = Path(
        args.root_path, "_".join([
            "-".join([str(num) for num in dataset])
            for dataset in [args.train_dataset, args.valid_dataset, args.test_dataset]
            if (dataset is not None and dataset != [])
        ])).resolve()
    logger.info(f"\n files is saved in {file_root_path}")

    pipeline_planer = PipelinePlaner.from_config_file(
        f"{Path(args.root_path).resolve()}/{args.tune_mode}_tuning_config.yaml")

    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

    eval_func = eval_acc

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

            data = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                             train_dataset=args.train_dataset, data_dir=args.data_dir,
                                             val_size=args.val_size).load_data()

            kwargs = {tune_mode: dict(wandb.config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            if run_idx == 0:
                print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")

            preprocessing_pipeline(data)

            set_split(data, data.train_idx, data.val_idx, data.test_idx)
            g = data.data.uns['HeteronetGraph']
            ref_data_name = f"{args.species}_{args.tissue}_{args.train_dataset}"
            dataset_ind, dataset_ood_tr, dataset_ood_te, adata = convert_dgl_to_original_format(
                g, data.data, ref_data_name)

            if len(dataset_ind.y.shape) == 1:
                dataset_ind.y = dataset_ind.y.unsqueeze(1)
            if len(dataset_ood_tr.y.shape) == 1:
                dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
            if isinstance(dataset_ood_te, list):
                for single_dataset_ood_te in dataset_ood_te:
                    if len(single_dataset_ood_te.y.shape) == 1:
                        single_dataset_ood_te.y = single_dataset_ood_te.y.unsqueeze(1)
            else:
                if len(dataset_ood_te.y.shape) == 1:
                    dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

            c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
            d = dataset_ind.graph['node_feat'].shape[1]

            model = scHeteroNet(d, c, dataset_ind.edge_index.to(device), dataset_ind.num_nodes,
                                hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout,
                                use_bn=args.use_bn, device=device, min_loss=100000)
            criterion = nn.NLLLoss()
            model.train()
            model.reset_parameters()
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            for epoch in range(args.epochs):
                loss = model.fit(dataset_ind, dataset_ood_tr, args.use_zinb, adata, args.zinb_weight, args.cl_weight,
                                 args.mask_ratio, criterion, optimizer)

            run_train_score = model.score(dataset_ind, dataset_ind.y, data.train_idx)
            run_valid_score = model.score(dataset_ind, dataset_ood_te.y, data.val_idx)
            run_test_score = model.score(dataset_ind, dataset_ind.y, data.test_idx)

            train_scores.append(run_train_score)
            valid_scores.append(run_valid_score)
            test_scores.append(run_test_score)

            logger.info(f"Run {run_idx + 1} finished. Valid Acc: {run_valid_score:.4f}, Test Acc: {run_test_score:.4f}")

            del model, data, dataset_ind, dataset_ood_tr, dataset_ood_te, g, adata
            gc.collect()
            if args.gpu != -1:
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
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)

    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(
            result_load_path=f"{args.summary_file_path}",
            step2_pipeline_planer=pipeline_planer,
            conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
            root_path=file_root_path,
            required_funs=["SaveRaw", "UpdateSizeFactors", "UpdateRaw", "HeteronetGraph", "SetConfig"],
            required_indexes=[1, 3, 5, 7, sys.maxsize],
        )
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
