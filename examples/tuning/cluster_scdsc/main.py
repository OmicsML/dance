import argparse
import pprint
import sys
from pathlib import Path

import numpy as np

import wandb
from dance import logger
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdsc import ScDSC
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1)]
    model_para = [512, 256, 256]
    # Cluster_para = [n_z1, n_z2, n_z3, n_init, n_input, n_clusters]
    Cluster_para = [256, 128, 32, 20, 100, 10]
    # Balance_para = [binary_crossentropy_loss, ce_loss, re_loss, zinb_loss, sigma]
    Balance_para = [1, 0.01, 0.1, 0.1, 1]

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument(
        "--dataset", type=str, default="worm_neuron_cell", choices=[
            "10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell", "human_pbmc2_cell",
            "mouse_kidney_cl2", "mouse_kidney_drop"
        ])
    # TODO: implement callbacks for "heat_kernel" and "cosine_normalized"
    parser.add_argument("--method", type=str, default="correlation", choices=["cosine", "correlation"])
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--n_enc_1", default=model_para[0], type=int)
    parser.add_argument("--n_enc_2", default=model_para[1], type=int)
    parser.add_argument("--n_enc_3", default=model_para[2], type=int)
    parser.add_argument("--n_dec_1", default=model_para[2], type=int)
    parser.add_argument("--n_dec_2", default=model_para[1], type=int)
    parser.add_argument("--n_dec_3", default=model_para[0], type=int)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n_z1", default=Cluster_para[0], type=int)
    parser.add_argument("--n_z2", default=Cluster_para[1], type=int)
    parser.add_argument("--n_z3", default=Cluster_para[2], type=int)
    parser.add_argument("--n_input", type=int, default=Cluster_para[4])
    parser.add_argument("--n_clusters", type=int, default=Cluster_para[5])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--v", type=int, default=1)
    parser.add_argument("--nb_genes", type=int, default=2000)
    parser.add_argument("--binary_crossentropy_loss", type=float, default=Balance_para[0])
    parser.add_argument("--ce_loss", type=float, default=Balance_para[1])
    parser.add_argument("--re_loss", type=float, default=Balance_para[2])
    parser.add_argument("--zinb_loss", type=float, default=Balance_para[3])
    parser.add_argument("--sigma", type=float, default=Balance_para[4])
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
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
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # inputs: adj, x, x_raw, n_counts
        inputs, y = data.get_data(return_type="default")
        args.n_input = inputs[1].shape[1]
        n_clusters = len(np.unique(y))

        model = ScDSC(
            pretrain_path=f"scdsc_{args.dataset}_pre.pkl",
            sigma=args.sigma,
            n_enc_1=args.n_enc_1,
            n_enc_2=args.n_enc_2,
            n_enc_3=args.n_enc_3,
            n_dec_1=args.n_dec_1,
            n_dec_2=args.n_dec_2,
            n_dec_3=args.n_dec_3,
            n_z1=args.n_z1,
            n_z2=args.n_z2,
            n_z3=args.n_z3,
            n_clusters=n_clusters,  #args.n_clusters,
            n_input=args.n_input,
            v=args.v,
            device=args.device)

        # Build and train model
        model.fit(inputs, y, lr=args.lr, epochs=args.epochs, bcl=args.binary_crossentropy_loss, cl=args.ce_loss,
                  rl=args.re_loss, zl=args.zinb_loss, pt_epochs=args.pretrain_epochs, pt_batch_size=args.batch_size,
                  pt_lr=args.pretrain_lr)

        score = model.score(None, y)
        wandb.log({"acc": score})
        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path, required_funs=["NeighborGraph", "SetConfig"],
                       required_indexes=[sys.maxsize - 1, sys.maxsize], metric="acc")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""Reproduction information
10X PBMC:
python scdsc.py --dataset 10X_PBMC --sigma 0.5 --topk 10 --pretrain_epochs 100 --v 3 --n_enc_1 1024 --n_enc_3 64 --n_dec_1 64 --n_z1 64

Mouse Bladder:
python scdsc.py --dataset mouse_bladder_cell --sigma 0.5 --topk 50 --pretrain_epochs 100 --v 7

Mouse ES:
python scdsc.py --dataset mouse_ES_cell --sigma 0.1 --topk 10 --pretrain_epochs 50 --v 2

Worm Neuron:
python scdsc.py --dataset worm_neuron_cell --sigma 0.5 --topk 10 --pretrain_epochs 100 --v 3 --n_enc_3 64 --n_dec_1 64 --n_z1 64 --n_z2 64
"""
