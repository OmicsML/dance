import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

import wandb
from dance.data.base import Data
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.EfNST import EfNsSTRunner
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151507",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    parser.add_argument("--sample_file", type=str, default=None)
    parser.add_argument("--sample_obs", type=int, default=None)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    args = parser.parse_args()
    file_root_path = Path(args.root_path, args.sample_number).resolve()
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        try:
            EfNST = EfNsSTRunner(
                platform=args.platform,
                pre_epochs=args.pre_epochs,  #### According to your own hardware, choose the number of training
                epochs=args.epochs,
                cnnType=args.cnnType,
                Conv_type=args.Conv_type,
                random_state=args.seed)
            dataloader = SpatialLIBDDataset(data_id=args.sample_number)
            data = dataloader.load_data(transform=None, cache=args.cache)
            if args.sample_obs is not None:
                data = Data(sc.pp.subsample(data.data, copy=True, n_obs=args.sample_obs, random_state=args.seed))
            # preprocessing_pipeline = EfNsSTRunner.preprocessing_pipeline(data_name=args.sample_number,verbose=args.verbose,cnnType=args.cnnType,
            #                                                              pca_n_comps=args.pca_n_comps,distType=args.distType,k=args.k,
            #                                                              dim_reduction=not args.no_dim_reduction,
            #                                                              min_cells=args.min_cells,platform=args.platform)
            kwargs = {tune_mode: dict(wandb.config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
            preprocessing_pipeline(data)

            (x, adj), y = data.get_data()
            adata = data.data
            adata = EfNST.fit(adata, x, graph_dict=adj, pretrain=args.pretrain)
            n_domains = len(np.unique(y))
            adata = EfNST._get_cluster_data(adata, n_domains=n_domains, priori=True)
            y_pred = EfNST.predict(adata)
        finally:
            EfNST.delete_imgs(adata)
        test_score = EfNST.score(x, y)
        wandb.log({"ARI": test_score})
        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path,
                      additional_sweep_ids=args.additional_sweep_ids)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(
            result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
            conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
            root_path=file_root_path,
            required_funs=["EfNSTImageTransform", "EfNSTAugmentTransform", "EfNSTGraphTransform",
                           "SetConfig"], required_indexes=[0, 1, 2, sys.maxsize], metric="ARI")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""
python main.py --sample_number 151507 --count 3 >> 151507/out.log 2>&1 &

python main.py --sample_number 151507
"""
