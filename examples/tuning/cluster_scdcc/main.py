import argparse
import os
from pathlib import Path
import pprint
import sys

import numpy as np
import wandb

from dance import logger
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdcc import ScDCC
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.transforms.preprocess import generate_random_pair
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--label_cells", default=0.1, type=float)
    parser.add_argument("--label_cells_files", default="label_mouse_ES_cell.txt")
    parser.add_argument("--n_pairwise", default=0, type=int)
    parser.add_argument("--n_pairwise_error", default=0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--data_dir", default="../temp_data")
    parser.add_argument("--dataset", default="mouse_kidney_10x", type=str,
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell","mouse_lung_cell","mouse_kidney_10x","mouse_kidney_cell"])
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--pretrain_epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--pretrain_lr", default=0.001, type=float)
    parser.add_argument("--sigma", default=2.5, type=float, help="coefficient of Gaussian noise")
    parser.add_argument("--gamma", default=1., type=float, help="coefficient of clustering loss")
    parser.add_argument("--ml_weight", default=1., type=float, help="coefficient of must-link loss")
    parser.add_argument("--cl_weight", default=1., type=float, help="coefficient of cannot-link loss")
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.00001, type=float)
    parser.add_argument("--ae_weights", default=None)
    parser.add_argument("--ae_weight_file", default="AE_weights.pth.tar")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--encodeLayer", type=int, nargs='+', default=[256, 64])
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
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # inputs: x, x_raw, n_counts
        inputs, y = data.get_train_data()
        n_clusters = len(np.unique(y))
        in_dim = inputs[0].shape[1]

        # Generate random pairs
        if not os.path.exists(args.label_cells_files):
            indx = np.arange(len(y))
            np.random.shuffle(indx)
            label_cell_indx = indx[0:int(np.ceil(args.label_cells * len(y)))]
        else:
            label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

        if args.n_pairwise > 0:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num = generate_random_pair(y, label_cell_indx, args.n_pairwise,
                                                                                 args.n_pairwise_error)
            print("Must link paris: %d" % ml_ind1.shape[0])
            print("Cannot link paris: %d" % cl_ind1.shape[0])
            print("Number of error pairs: %d" % error_num)
        else:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])

        # Build and train moodel
        model = ScDCC(input_dim=in_dim, z_dim=args.z_dim, n_clusters=n_clusters, encodeLayer=args.encodeLayer,
                      decodeLayer=args.encodeLayer[::-1], sigma=args.sigma, gamma=args.gamma, ml_weight=args.ml_weight,
                      cl_weight=args.ml_weight, device=args.device, pretrain_path=f"scdcc_{args.dataset}_pre.pkl")
        model.fit(inputs, y, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, ml_ind1=ml_ind1,
                  ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2, update_interval=args.update_interval, tol=args.tol,
                  pt_batch_size=args.batch_size, pt_lr=args.pretrain_lr, pt_epochs=args.pretrain_epochs)

        # Evaluate model predictions
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
            required_funs=["SaveRaw","UpdateRaw","SetConfig"],
            required_indexes=[2, 5,sys.maxsize],
            metric="acc"
        )
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
    
""" Reproduction information
10X PBMC:
python scdcc.py --dataset 10X_PBMC --label_cells_files label_10X_PBMC.txt --pretrain_epochs 300 --epochs 100 --sigma 2 --n_pairwise 10000 --cache

Mouse ES:
python scdcc.py --dataset mouse_ES_cell --label_cells_files label_mouse_ES_cell.txt --pretrain_epochs 300 --epochs 100 --sigma 1.75 --encodeLayer 512 256  --n_pairwise 10000 --cache

Worm Neuron:
python scdcc.py --dataset worm_neuron_cell --label_cells_files label_worm_neuron_cell.txt --pretrain_epochs 300 --epochs 100 --n_pairwise 20000 --cache

Mouse Bladder:
python scdcc.py --dataset mouse_bladder_cell --label_cells_files label_mouse_bladder_cell.txt --pretrain_epochs 300 --epochs 100 --sigma 3.25 --n_pairwise 10000 --cache
"""
