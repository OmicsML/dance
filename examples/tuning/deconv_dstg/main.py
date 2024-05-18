import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import torch

import wandb
from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo import DSTG
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.AVAILABLE_DATA)
    parser.add_argument("--datadir", default="../temp_data", help="Directory to save the data.")
    parser.add_argument("--sc_ref", type=bool, default=True, help="Reference scRNA (True) or cell-mixtures (False).")
    parser.add_argument("--num_pseudo", type=int, default=500, help="Number of pseudo mixtures to generate.")
    parser.add_argument("--n_hvg", type=int, default=2000, help="Number of HVGs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--k_filter", type=int, default=200, help="Graph node filter.")
    parser.add_argument("--num_cc", type=int, default=30, help="Dimension of canonical correlation analysis.")
    parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
    parser.add_argument("--nhid", type=int, default=16, help="Number of neurons in latent layer.")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout rate.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train the model.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument("--data_dir", type=str, default='../temp_data', help='test directory')
    parser.add_argument("--device", default="auto", help="Computation device.")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    args = parser.parse_args()
    pprint(vars(args))
    file_root_path = Path(args.root_path, args.dataset).resolve()
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)

        # Load dataset
        dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
        data = dataset.load_data()
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)
        (adj, x), y = data.get_data(return_type="default")
        x, y = torch.FloatTensor(x), torch.FloatTensor(y.values)
        adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                                       torch.FloatTensor(adj.data.astype(np.int32)))
        train_mask = data.get_split_mask("pseudo", return_type="torch")
        inputs = (adj, x, train_mask)

        # Train and evaluate model
        model = DSTG(nhid=args.nhid, bias=args.bias, dropout=args.dropout, device=args.device)
        pred = model.fit_predict(inputs, y, lr=args.lr, max_epochs=args.epochs, weight_decay=args.wd)
        test_mask = data.get_split_mask("test", return_type="torch")
        score = model.default_score_func(y[test_mask], pred[test_mask])
        wandb.log({"MSE": score})

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path, ascending=True,
                       required_funs=["FilterGenesCommon", "PseudoMixture", "RemoveSplit", "DSTGraph",
                                      "SetConfig"], required_indexes=[0, 1, 2, sys.maxsize - 1,
                                                                      sys.maxsize], metric="MSE")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce DSTG benchmarks, please refer to command lines belows:

GSE174746:
$ python dstg.py --dataset GSE174746 --nhid 16 --lr .0001 --k_filter 50

CARD synthetic:
$ python dstg.py --dataset CARD_synthetic --nhid 16 --lr .001 --k_filter 50

SPOTLight synthetic:
$ python dstg.py --dataset SPOTLight_synthetic --nhid 32 --lr .1 --epochs 25

"""
