import argparse
import gc
import math
import os
import time
from pathlib import Path
import random

from PIL import Image
from efficientnet_pytorch import EfficientNet
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
import torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import BatchNorm, Sequential
from torchvision import transforms
import torchvision.transforms as transforms
from tqdm import tqdm

import wandb
from dance import logger
from dance.data.base import Data
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.EfNST import (
    EfNSTAugmentTransform,
    EfNSTConcatgTransform,
    EfNSTGraphTransform,
    EfNSTImageTransform,
    EfNsSTRunner,
)
from dance.pipeline import PipelinePlaner, save_summary_data
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA
from dance.transforms.filter import (
    FilterGenesPercentile,
    HighlyVariableGenesLogarithmizedByTopGenes,
)
from dance.transforms.misc import Compose, SetConfig
from dance.utils import set_seed,sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func
from torch_sparse import SparseTensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151507",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument("--sample_file", type=str, default=None)
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    parser.add_argument("--cnnType", type=str, default="efficientnet-b0")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain the model.")
    parser.add_argument("--pre_epochs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--Conv_type", type=str, default="ResGatedGraphConv")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information.")
    parser.add_argument("--pca_n_comps", type=int, default=200, help="Number of PCA components.")
    parser.add_argument("--distType", type=str, default="KDTree", help="Distance type.")
    parser.add_argument("--k", type=int, default=12, help="Number of neighbors.")
    parser.add_argument("--no_dim_reduction", action="store_true", help="Print detailed information.")
    parser.add_argument("--min_cells", type=int, default=3, help="Minimum number of cells.")
    parser.add_argument("--platform", type=str, default="Visium", help="Platform type.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0').")
    args = parser.parse_args()
    start_time = time.time()
    file_root_path = Path(args.root_path, args.sample_number).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{Path(args.root_path).resolve()}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))

        scores = []
        inner_scores = []

        # Start Timer

        for run_idx in range(args.num_runs):
            print(f"Starting Run {run_idx + 1}/{args.num_runs}")

            current_seed = args.seed + run_idx
            set_seed(current_seed, extreme_mode=True)

            try:
                # Initialize model
                EfNST = EfNsSTRunner(
                    platform=args.platform,
                    pre_epochs=args.pre_epochs,  #### According to your own hardware, choose the number of training
                    epochs=args.epochs,
                    cnnType=args.cnnType,
                    Conv_type=args.Conv_type,
                    random_state=current_seed)

                # Load data and perform necessary preprocessing
                dataloader = SpatialLIBDDataset(data_id=args.sample_number)
                kwargs = {tune_mode: dict(wandb.config)}
                preprocessing_pipeline = pipeline_planer.generate(**kwargs)
                if run_idx == 0:
                    print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
                data = dataloader.load_data(transform=None, cache=args.cache)
                sub_data(data.data)
                data.data.uns['data_name']=args.sample_number
                preprocessing_pipeline(data)
                (x, adj), y = data.get_data()
                adata = data.data

                # Fit the model and evaluate
                adata = EfNST.fit(adata, x, graph_dict=adj, pretrain=args.pretrain)
                n_domains = len(np.unique(y))
                adata = EfNST._get_cluster_data(adata, n_domains=n_domains, priori=True)
                y_pred = EfNST.predict(adata)

                # Calculate ARI score
                score = adjusted_rand_score(y, y_pred)

                # Calculate internal evaluation metrics
                silhouette_score = resolve_score_func("silhouette")
                calinski_harabasz_score = resolve_score_func("calinski_harabasz")
                davies_bouldin_score = resolve_score_func("davies_bouldin")
                run_inner_score = calculate_unified_scores({
                    "silhouette": silhouette_score(x, y_pred),
                    "calinski_harabasz": calinski_harabasz_score(x, y_pred),
                    "davies_bouldin": davies_bouldin_score(x, y_pred)
                })

                scores.append(score)
                inner_scores.append(run_inner_score)

                print(f"Run {run_idx + 1} finished. ARI: {score:.4f}, Inner Score: {run_inner_score:.4f}")

            finally:
                if "adata" in locals():
                    EfNST.delete_imgs(adata)
                del EfNST, data
                gc.collect()

        # Stop Timer
        total_time_seconds = time.time() - start_time

        avg_score = np.mean(scores)
        avg_inner_score = np.mean(inner_scores)

        # Calculate Speed Score and Combined Score
        speed_score = 1.0 / (1.0 + total_time_seconds / 300.0)
        combined_score = 0.8 * avg_inner_score + 0.2 * speed_score

        print(f"Averaged over {args.num_runs} runs - ARI: {avg_score:.4f}, Inner Score: {avg_inner_score:.4f}, Time: {total_time_seconds:.2f}s, Combined Score: {combined_score:.4f}")

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
""" To reproduce EfNST on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151507:
$ python EfNST.py --sample_number 151507 --tune_mode params

human dorsolateral prefrontal cortex sample 151673:
$ python EfNST.py --sample_number 151673 --tune_mode params

human dorsolateral prefrontal cortex sample 151676:
$ python EfNST.py --sample_number 151676 --tune_mode params
"""
