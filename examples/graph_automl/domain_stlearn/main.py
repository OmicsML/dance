import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from dance import logger
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stlearn import StKmeans, StLouvain
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func

MODES = ["louvain", "kmeans"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--mode", type=str, default="louvain", choices=MODES)
    parser.add_argument("--n_clusters", type=int, default=17, help="the number of clusters")
    parser.add_argument("--n_components", type=int, default=50, help="the number of components in PCA")
    parser.add_argument("--device", type=str, default="cuda", help="device for resnet extract feature")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    parser.add_argument("--sample_file", type=str, default=None)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    args = parser.parse_args()
    file_root_path = Path(args.root_path, args.sample_number).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(
        f"{Path(args.root_path).resolve()}/{args.tune_mode}_tuning_config.yaml")

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
            if args.mode == "kmeans":
                model = StKmeans(n_clusters=args.n_clusters)
            elif args.mode == "louvain":
                model = StLouvain(resolution=0.6)
            else:
                raise ValueError(f"Unknown mode {args.mode!r}, available options are {MODES}")

            # Load data and perform necessary preprocessing
            dataloader = SpatialLIBDDataset(data_id=args.sample_number, data_dir=args.data_dir,
                                            sample_file=args.sample_file)
            data = dataloader.load_data(cache=args.cache)
            # Prepare preprocessing pipeline and apply it to data
            kwargs = {tune_mode: dict(wandb.config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            if run_idx == 0:
                print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
            sub_data(data.data)
            preprocessing_pipeline(data)
            x, y = data.get_data(return_type="default")

            # Train and evaluate model
            score = model.fit_score(x, y.values.ravel())

            # Get predictions for internal evaluation metrics
            pred = model.predict(x)

            # Calculate internal evaluation metrics
            silhouette_score = resolve_score_func("silhouette")
            calinski_harabasz_score = resolve_score_func("calinski_harabasz")
            davies_bouldin_score = resolve_score_func("davies_bouldin")

            # 检查 x 是否为稀疏数组，如果是则转换为稠密
            if hasattr(x, 'toarray'):
                x = x.toarray()
            elif hasattr(x, 'todense'):
                x = x.todense()
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

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path,
                      additional_sweep_ids=args.additional_sweep_ids)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path,
                       required_funs=["MorphologyFeatureCNN", "SMEGraph", "SMEFeature", "NeighborGraph",
                                      "SetConfig"], required_indexes=[4, 6, 7, 8, sys.maxsize], metric="ARI")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
""" To reproduce stlearn on other samples, please refer to command lines belows:
NOTE: since the stlearn method is unstable, you have to run multiple times to get
      best performance.

human dorsolateral prefrontal cortex sample 151673:
$ python stlearn.py --n_clusters 20 --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python stlearn.py --n_clusters 20 --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python stlearn.py --n_clusters 20 --sample_number 151507
"""
