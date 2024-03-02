import argparse
import pprint
import sys
from pathlib import Path
from typing import get_args

import numpy as np
import torch

import wandb
from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.pipeline import PipelinePlaner, get_params, get_step3_yaml, run_step3, save_summary_data
from dance.typing import LogLevel
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    # parser.add_argument("--dense_dim", type=int, default=400, help="number of hidden gcn units")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--hidden_dim", type=int, default=200, help="number of hidden gcn units")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--valid_dataset", nargs="+", default=[1970], help="List of valid dataset ids.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")
    file_root_path = Path(
        args.root_path, "_".join([
            "/".join([str(num) for num in dataset])
            for dataset in [args.train_dataset, args.valid_dataset, args.test_dataset]
        ])).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Load data and perform necessary preprocessing
        data = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                         train_dataset=args.train_dataset, valid_dataset=args.valid_dataset,
                                         data_dir="./temp_data").load_data()
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # Obtain training and testing data
        y_train = data.get_y(split_name="train", return_type="torch").argmax(1)
        y_valid = data.get_y(split_name="val", return_type="torch")
        y_test = data.get_y(split_name="test", return_type="torch")

        # Get cell feature graph for scDeepSort
        # TODO: make api for the following block?
        g = data.data.uns["CellFeatureGraph"]
        num_genes = data.shape[1]
        # Initialize model and get model specific preprocessing pipeline
        dense_dim = g.ndata["features"].shape[1]
        model = ScDeepSort(dense_dim, args.hidden_dim, args.n_layers, args.species, args.tissue, dropout=args.dropout,
                           batch_size=args.batch_size, device=args.device)

        gene_ids = torch.arange(num_genes)
        train_cell_ids = torch.LongTensor(data.train_idx) + num_genes
        valid_cell_ids = torch.LongTensor(data.val_idx) + num_genes
        test_cell_ids = torch.LongTensor(data.test_idx) + num_genes
        g_train = g.subgraph(torch.concat((gene_ids, train_cell_ids)))
        g_valid = g.subgraph(torch.concat((gene_ids, valid_cell_ids)))
        g_test = g.subgraph(torch.concat((gene_ids, test_cell_ids)))

        # Train and evaluate the model
        model.fit(g_train, y_train, epochs=args.n_epochs, lr=args.lr, weight_decay=args.weight_decay,
                  val_ratio=args.test_rate)
        score = model.score(g_valid, y_valid)
        test_score = model.score(g_test, y_test)
        wandb.log({"acc": score, "test_acc": test_score})
        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
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

Mouse Brain
$ python scdeepsort.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python scdeepsort.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python scdeepsort.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
