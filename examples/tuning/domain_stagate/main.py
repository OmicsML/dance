import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

import wandb
from dance import logger
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stagate import Stagate
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--hidden_dims", type=list, default=[512, 32], help="hidden dimensions")
    parser.add_argument("--rad_cutoff", type=int, default=150, help="")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--high_variable_genes", type=int, default=3000, help="")
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument("--sample_file", type=str, default=None)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    args = parser.parse_args()
    file_root_path = Path(args.root_path, args.sample_number).resolve()
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Initialize model and get model specific preprocessing pipeline
        model = Stagate([args.high_variable_genes] + args.hidden_dims)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number, data_dir=args.data_dir,
                                        sample_file=args.sample_file)
        data = dataloader.load_data(cache=args.cache)
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)
        (x, adj), y = data.get_data(return_type="default")
        if not isinstance(x, np.ndarray):
            x = x.A
        edge_list_array = np.vstack(np.nonzero(adj))

        # Train and evaluate model
        model = Stagate([x.shape[1]] + args.hidden_dims)
        score = model.fit_score((x, edge_list_array), y, epochs=args.epochs, random_state=args.seed)
        wandb.log({"ARI": score})
        gc.collect()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path, required_funs=["StagateGraph", "SetConfig"],
                       required_indexes=[4, sys.maxsize], metric="ARI")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
""" To reproduce Stagate on other samples, please refer to command lines belows:
NOTE: since the stagate method is unstable, you have to run at least 5 times to get
      best performance. (same with original Stagate paper)

human dorsolateral prefrontal cortex sample 151673:
$ python stagate.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python stagate.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python stagate.py --sample_number 151507
"""
