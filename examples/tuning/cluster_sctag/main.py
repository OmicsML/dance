import argparse
import os
import pprint
import sys
from pathlib import Path

import numpy as np

import wandb
from dance import logger
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.sctag import ScTAG
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default="../temp_data", type=str)
    parser.add_argument(
        "--dataset", default="mouse_bladder_cell", type=str, choices=[
            "10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell", "mouse_kidney_cl2",
            "mouse_kidney_10x", "mouse_lung_cell", "mouse_kidney_drop", "mouse_kidney_cell", "human_pbmc2_cell",
            "human_skin_cell"
        ])
    parser.add_argument("--k_neighbor", default=15, type=int)
    parser.add_argument("--highly_genes", default=3000, type=int)
    parser.add_argument("--pca_dim", default=50, type=int)
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--latent_dim", default=15, type=int)
    parser.add_argument("--dec_dim", default=None, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--pretrain_epochs", default=200, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--w_a", default=1, type=float)
    parser.add_argument("--w_x", default=1, type=float)
    parser.add_argument("--w_d", default=0, type=float)
    parser.add_argument("--w_c", default=1, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--min_dist", default=0.5, type=float)
    parser.add_argument("--max_dist", default=20.0, type=float)
    parser.add_argument("--info_step", default=50, type=int)
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    args = parser.parse_args()
    logger.info(f"\n{pprint.pformat(vars(args))}")
    file_root_path = Path(args.root_path, args.dataset).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Load data and perform necessary preprocessing
        dataloader = ClusteringDataset(args.data_dir, args.dataset)
        data = dataloader.load_data(cache=args.cache)
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # inputs: adj, x, x_raw, n_counts
        inputs, y = data.get_train_data()
        n_clusters = len(np.unique(y))

        # Build and train model
        model = ScTAG(n_clusters=n_clusters, k=args.k, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                      dec_dim=args.dec_dim, dropout=args.dropout, device=args.device, alpha=args.alpha,
                      pretrain_path=f"sctag_{args.dataset}_pre.pkl")
        model.fit(inputs, y, epochs=args.epochs, pretrain_epochs=args.pretrain_epochs, lr=args.lr, w_a=args.w_a,
                  w_x=args.w_x, w_c=args.w_c, w_d=args.w_d, info_step=args.info_step, max_dist=args.max_dist,
                  min_dist=args.min_dist)

        # Evaluate model predictions
        score = model.score(None, y)
        wandb.log({"acc": score})
        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path, required_funs=["SaveRaw", "UpdateRaw", "NeighborGraph", "SetConfig"],
                       required_indexes=[2, 5, sys.maxsize - 1, sys.maxsize], metric="acc")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""Reproduction information
10X PBMC:
python sctag.py --dataset 10X_PBMC --pretrain_epochs 100 --w_a 0.01 --w_x 3 --w_c 0.1 --dropout 0.5

Mouse ES:
python sctag.py --dataset mouse_ES_cell --pretrain_epochs 100 --w_a 0.01 --w_x 0.75 --w_c 1

Worm Neuron:
python sctag.py --dataset worm_neuron_cell --w_a 0.01 --w_x 2 --w_c 0.25 --k 1

Mouse Bladder:
python sctag.py --dataset mouse_bladder_cell --pretrain_epochs 100 --w_a 0.1 --w_x 2.5 --w_c 3
"""
