import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import wandb

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.spotlight import SPOTlight
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.AVAILABLE_DATA)
    parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--rank", type=int, default=2, help="Rank of the NMF module.")
    parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
    parser.add_argument("--max_iter", type=int, default=4000, help="Maximum optimization iteration.")
    parser.add_argument("--device", default="auto", help="Computation device.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    args = parser.parse_args()
    pprint(vars(args))
    file_root_path = Path(args.root_path, args.dataset).resolve()
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Load dataset
        preprocessing_pipeline = SPOTlight.preprocessing_pipeline()
        dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
        data = dataset.load_data(cache=args.cache)
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)
        cell_types = data.data.obsm["cell_type_portion"].columns.tolist()

        x, y = data.get_data(split_name="test", return_type="torch")
        ref_count = data.get_feature(split_name="ref", return_type="numpy")
        ref_annot = data.get_feature(split_name="ref", return_type="numpy", channel="cellType", channel_type="obs")

        # Train and evaluate model
        model = SPOTlight(ref_count, ref_annot, cell_types, rank=args.rank, bias=args.bias, device=args.device)
        score = model.fit_score(x, y, lr=args.lr, max_iter=args.max_iter)
        wandb.log({"MSE": score})

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path, ascending=True, required_funs=["SetConfig"],
                       required_indexes=[sys.maxsize], metric="MSE")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce SpatialDecon benchmarks, please refer to command lines belows:

GSE174746:
$ python spotlight.py --dataset GSE174746 --lr .1 --max_iter 15000 --rank 4 --bias 0

CARD synthetic:
$ python spotlight.py --dataset CARD_synthetic --lr .1 --max_iter 100 --rank 8 --bias 0

SPOTLight synthetic:
$ python spotlight.py --dataset SPOTLight_synthetic --lr .1 --max_iter 150 --rank 10 --bias 0

"""
