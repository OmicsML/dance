import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import wandb

from dance import logger
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN, refine
from dance.pipeline import PipelinePlaner, save_summary_data
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--beta", type=int, default=49, help="")
    parser.add_argument("--alpha", type=int, default=1, help="")
    parser.add_argument("--p", type=float, default=0.05,
                        help="percentage of total expression contributed by neighborhoods.")
    parser.add_argument("--l", type=float, default=0.5, help="the parameter to control percentage p.")
    parser.add_argument("--start", type=float, default=0.01, help="starting value for searching l.")
    parser.add_argument("--end", type=float, default=1000, help="ending value for searching l.")
    parser.add_argument("--tol", type=float, default=5e-3, help="tolerant value for searching l.")
    parser.add_argument("--max_run", type=int, default=200, help="max runs.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--n_clusters", type=int, default=7, help="the number of clusters")
    parser.add_argument("--step", type=float, default=0.1, help="")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--device", default="cpu", help="Computation device.")
    parser.add_argument("--seed", type=int, default=100, help="")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--tune_mode", default="params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument("--sample_file", type=str, default=None)
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    args = parser.parse_args()
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
        start_time = time.time()

        for run_idx in range(args.num_runs):
            print(f"Starting Run {run_idx + 1}/{args.num_runs}")

            current_seed = args.seed + run_idx
            set_seed(current_seed)

            # Initialize model and get model specific preprocessing pipeline
            model = SpaGCN(device=args.device)
            kwargs = {tune_mode: dict(wandb.config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            if run_idx == 0:
                print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")

            # Load data and perform necessary preprocessing
            dataloader = SpatialLIBDDataset(data_id=args.sample_number)
            data = dataloader.load_data(transform=None, cache=args.cache)
            sub_data(data.data)
            preprocessing_pipeline(data)
            (x, adj, adj_2d), y = data.get_train_data()

            # Train and evaluate model
            l = model.search_l(args.p, adj, start=args.start, end=args.end, tol=args.tol, max_run=args.max_run)
            model.set_l(l)
            res = model.search_set_res((x, adj), l=l, target_num=args.n_clusters, start=0.4, step=args.step,
                                       tol=args.tol, lr=args.lr, epochs=args.epochs, max_run=args.max_run)

            model.fit((x, adj), init_spa=True, init="louvain", tol=args.tol, lr=args.lr, epochs=args.epochs, res=res)
            embed, pred = model.predict((x, adj), return_embed=True)
            score = model.default_score_func(y, pred)

            refined_pred = refine(sample_id=data.data.obs_names.tolist(), pred=pred.tolist(), dis=adj_2d,
                                  shape="hexagon")
            score_refined = model.default_score_func(y, refined_pred)

            # Calculate internal evaluation metrics
            silhouette_score = resolve_score_func("silhouette")
            calinski_harabasz_score = resolve_score_func("calinski_harabasz")
            davies_bouldin_score = resolve_score_func("davies_bouldin")
            run_inner_score = calculate_unified_scores({
                "silhouette": silhouette_score(embed, refined_pred),
                "calinski_harabasz": calinski_harabasz_score(embed, refined_pred),
                "davies_bouldin": davies_bouldin_score(embed, refined_pred)
            })

            scores.append(score_refined)
            inner_scores.append(run_inner_score)

            print(f"Run {run_idx + 1} finished. ARI: {score_refined:.4f}, Inner Score: {run_inner_score:.4f}")

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
""" To reproduce SpaGCN on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spagcn.py --sample_number 151673 --lr 0.1

human dorsolateral prefrontal cortex sample 151676:
$ python spagcn.py --sample_number 151676 --lr 0.02

human dorsolateral prefrontal cortex sample 151507:
$ python spagcn.py --sample_number 151507 --lr 0.009
"""
