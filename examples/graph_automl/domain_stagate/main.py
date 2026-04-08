import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import wandb

from dance import logger
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stagate import Stagate
from dance.pipeline import PipelinePlaner, save_summary_data
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--hidden_dims", type=list, default=[512, 32], help="hidden dimensions")
    parser.add_argument("--rad_cutoff", type=int, default=150, help="")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--high_variable_genes", type=int, default=3000, help="")
    parser.add_argument("--seed", type=int, default=3, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument("--sample_file", type=str, default=None)
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0').")
    args = parser.parse_args()
    start_time = time.time()

    file_root_path = Path(args.root_path, args.sample_number).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(
        f"{Path(args.root_path).resolve()}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))

        scores = []
        inner_scores = []

        # Start Timer

        for run_idx in range(args.num_runs):
            print(f"Starting Run {run_idx + 1}/{args.num_runs}")

            current_seed = args.seed + run_idx
            set_seed(current_seed)

            # Initialize model and get model specific preprocessing pipeline
            kwargs = {tune_mode: dict(wandb.config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            if run_idx == 0:
                print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")

            # Load data and perform necessary preprocessing
            dataloader = SpatialLIBDDataset(data_id=args.sample_number)
            data = dataloader.load_data(transform=None, cache=args.cache)
            sub_data(data.data)
            preprocessing_pipeline(data)
            adj, y = data.get_data(return_type="default")

            x = data.data.X.A if sp.issparse(data.data.X) else data.data.X
            edge_list_array = np.vstack(np.nonzero(adj))

            model = Stagate([min(args.high_variable_genes, x.shape[1])] + args.hidden_dims, device=args.device)

            # Train and evaluate model
            score = model.fit_score((x, edge_list_array), y, epochs=args.epochs, random_state=current_seed)
            pred = model.predict()

            # Calculate internal evaluation metrics
            silhouette_score = resolve_score_func("silhouette")
            calinski_harabasz_score = resolve_score_func("calinski_harabasz")
            davies_bouldin_score = resolve_score_func("davies_bouldin")
            run_inner_score = calculate_unified_scores({
                "silhouette": silhouette_score(x, pred),
                "calinski_harabasz": calinski_harabasz_score(x, pred),
                "davies_bouldin": davies_bouldin_score(x, pred)
            })

            scores.append(score)
            inner_scores.append(run_inner_score)

            print(f"Run {run_idx + 1} finished. ARI: {score:.4f}, Inner Score: {run_inner_score:.4f}")

            del model, data
            gc.collect()

        # Stop Timer
        total_time_seconds = time.time() - start_time

        avg_score = np.mean(scores)
        avg_inner_score = np.mean(inner_scores)

        # Calculate Speed Score and Combined Score
        speed_score = 1.0 / (1.0 + total_time_seconds / 300.0)
        combined_score = 0.8 * avg_inner_score + 0.2 * speed_score

        print(
            f"Averaged over {args.num_runs} runs - ARI: {avg_score:.4f}, Inner Score: {avg_inner_score:.4f}, Time: {total_time_seconds:.2f}s, Combined Score: {combined_score:.4f}"
        )

        wandb.log({
            "ARI": avg_score,
            "inner_score": avg_inner_score,
            "time": total_time_seconds,
            "speed_score": speed_score,
            "combined_score": combined_score
        })

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(evaluate_pipeline, sweep_id=args.sweep_id,
                                                                  count=args.count)
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path,
                      additional_sweep_ids=args.additional_sweep_ids)
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
