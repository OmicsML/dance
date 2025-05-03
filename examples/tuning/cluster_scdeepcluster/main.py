import argparse
import os
import pprint
import sys
from pathlib import Path

import numpy as np
import wandb

from dance import logger
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdeepcluster import ScDeepCluster
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--knn", default=20, type=int,
                        help="number of nearest neighbors, used by the Louvain algorithm")
    parser.add_argument(
        "--resolution", default=.8, type=float,
        help="resolution parameter, used by the Louvain algorithm, larger value for more number of clusters")
    parser.add_argument("--select_genes", default=0, type=int, help="number of selected genes, 0 means using all genes")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--data_dir", default="../temp_data")
    parser.add_argument(
        "--dataset", default="human_pbmc2_cell", type=str, choices=[
            "10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell", "human_pbmc2_cell",
            "mouse_kidney_cell", "human_skin_cell", "mouse_kidney_drop", "mouse_kidney_cl2"
        ])
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--pretrain_epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--pretrain_lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=1., type=float, help="coefficient of clustering loss")
    parser.add_argument("--sigma", default=2.5, type=float, help="coefficient of random noise")
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.001, type=float,
                        help="tolerance for delta clustering labels to terminate training stage")
    parser.add_argument("--ae_weights", default=None, help="file to pretrained weights, None for a new pretraining")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--encodeLayer", type=int, nargs='+', default=[256, 64])
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
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # inputs: x, x_raw, n_counts
        inputs, y = data.get_train_data()
        n_clusters = len(np.unique(y))
        in_dim = inputs[0].shape[1]

        # Build and train model
        model = ScDeepCluster(input_dim=in_dim, z_dim=args.z_dim, encodeLayer=args.encodeLayer,
                              decodeLayer=args.encodeLayer[::-1], sigma=args.sigma, gamma=args.gamma,
                              device=args.device, pretrain_path=f"scdeepcluster_{args.dataset}_pre.pkl")
        model.fit(inputs, y, n_clusters=n_clusters, y_pred_init=None, lr=args.lr, batch_size=args.batch_size,
                  epochs=args.epochs, update_interval=args.update_interval, tol=args.tol, pt_batch_size=args.batch_size,
                  pt_lr=args.pretrain_lr, pt_epochs=args.pretrain_epochs)

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
                       root_path=file_root_path, required_funs=["SaveRaw", "UpdateRaw", "SetConfig"],
                       required_indexes=[2, 5, sys.maxsize], metric="acc")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
""" Reproduction information
10X PBMC:
python scdeepcluster.py --dataset 10X_PBMC --pretrain_epochs 300 --epochs 100 --sigma 2

Mouse ES:
python scdeepcluster.py --dataset mouse_ES_cell --pretrain_epochs 300 --epochs 100 --sigma 1.75 --encodeLayer 512 256

Worm Neuron:
python scdeepcluster.py --dataset worm_neuron_cell --pretrain_epochs 300 --epochs 100 --sigma 1.5

Mouse Bladder:
python scdeepcluster.py --dataset mouse_bladder_cell --pretrain_epochs 300 --sigma 2 --epochs 100
"""
