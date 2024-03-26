import argparse
import pprint
import sys
from pathlib import Path

import numpy as np

import wandb
from dance import logger
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.graphsc import GraphSC
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-dv", "--device", default="auto")
    # parser.add_argument("-if", "--in_feats", default=50, type=int)
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
    parser.add_argument("-nw", "--normalize_weights", default="log_per_cell", choices=["log_per_cell", "per_cell"])
    parser.add_argument("-ac", "--activation", default="relu", choices=["leaky_relu", "relu", "prelu", "gelu"])
    parser.add_argument("-drop", "--dropout", default=0.1, type=float)
    parser.add_argument("-nf", "--node_features", default="scale", choices=["scale_by_cell", "scale", "none"])
    parser.add_argument("-sev", "--same_edge_values", default=False, action="store_true")
    parser.add_argument("-en", "--edge_norm", default=True, action="store_true")
    parser.add_argument("-hr", "--hidden_relu", default=False, action="store_true")
    parser.add_argument("-hbn", "--hidden_bn", default=False, action="store_true")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-nl", "--n_layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("-agg", "--agg", default="sum", choices=["sum", "mean"])
    parser.add_argument("-hd", "--hidden_dim", type=int, default=200)
    parser.add_argument("-nh", "--n_hidden", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("-h1", "--hidden_1", type=int, default=300)
    parser.add_argument("-h2", "--hidden_2", type=int, default=0)
    parser.add_argument("-ng", "--nb_genes", type=int, default=3000)
    parser.add_argument("-nr", "--num_run", type=int, default=1)
    parser.add_argument("-nbw", "--num_workers", type=int, default=1)
    parser.add_argument("-eve", "--eval_epoch", action="store_true")
    parser.add_argument("-show", "--show_epoch_ari", action="store_true")
    parser.add_argument("-plot", "--plot", default=False, action="store_true")
    parser.add_argument("-dd", "--data_dir", default="./temp_data", type=str)
    parser.add_argument("-data", "--dataset", default="worm_neuron_cell",
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell","mouse_kidney_10x","human_ILCS_cell","mouse_kidney_drop","mouse_lung_cell"])
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    args = parser.parse_args()
    aris = []
    logger.info(f"\n{pprint.pformat(vars(args))}")
    file_root_path = Path(args.root_path, args.dataset).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

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

        graph, y = data.get_train_data()
        n_clusters = len(np.unique(y))

        g = data.data.uns["CellFeatureGraph"]
        in_feats = g.ndata["features"].shape[1]

        # Evaluate model for several runs
        # for run in range(args.num_run):
        # set_seed(args.seed + run)
        model = GraphSC(agg=args.agg, activation=args.activation, in_feats=in_feats, n_hidden=args.n_hidden,
                        hidden_dim=args.hidden_dim, hidden_1=args.hidden_1, hidden_2=args.hidden_2,
                        dropout=args.dropout, n_layers=args.n_layers, hidden_relu=args.hidden_relu,
                        hidden_bn=args.hidden_bn, n_clusters=n_clusters, cluster_method="leiden",
                        num_workers=args.num_workers, device=args.device)
        model.fit(graph, epochs=args.epochs, lr=args.learning_rate, show_epoch_ari=args.show_epoch_ari,
                  eval_epoch=args.eval_epoch)
        score = model.score(None, y)
        wandb.log({"acc": score})
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
            metric="acc"
        )
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
    
""" Reproduction information
10X PBMC:
python graphsc.py --dataset 10X_PBMC --dropout 0.5

Mouse ES:
python graphsc.py --dataset mouse_ES_cell

Worm Neuron:
python graphsc.py --dataset worm_neuron_cell

Mouse Bladder:
python graphsc.py --dataset mouse_bladder_cell
"""
