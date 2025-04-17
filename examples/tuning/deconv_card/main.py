import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np

import wandb
from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.card import Card
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.AVAILABLE_DATA)
    parser.add_argument("--datadir", default="../temp_data", help="Directory to save the data.")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum optimization iteration.")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Optimization threshold.")
    parser.add_argument("--location_free", action="store_true", help="Do not supply spatial location if set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    args = parser.parse_args()
    file_root_path = Path(args.root_path, args.dataset).resolve()
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Load dataset
        dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
        data = dataset.load_data()
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)
        # inputs: x_count, x_spatial
        inputs, y = data.get_data(split_name="test", return_type="numpy")
        basis = data.get_feature(return_type="default", channel="CellTopicProfile",
                                 channel_type="varm")  #降维之后没法寻找特定基因的varm了，也就是基因在每个细胞类型上的表现

        # Train and evaluate model
        model = Card(basis, random_state=args.seed)
        score = model.fit_score(inputs, y, max_iter=args.max_iter, epsilon=args.epsilon,
                                location_free=args.location_free)
        wandb.log({"MSE": score})

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path,
                       required_funs=["CellTopicProfile", "FilterGenesCommon", "FilterGenesMarker",
                                      "SetConfig"], required_indexes=[0, 2, 3,
                                                                      sys.maxsize], metric="MSE", ascending=True)
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce CARD benchmarks, please refer to command lines belows:

GSE174746:
$ python card.py --dataset GSE174746 --location_free

CARD synthetic:
$ python card.py --dataset CARD_synthetic

SPOTLight synthetic:
$ python card.py --dataset SPOTLight_synthetic --location_free

"""
