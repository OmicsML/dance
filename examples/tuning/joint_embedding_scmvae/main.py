import argparse
import os
import pprint
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import wandb
from sklearn import preprocessing

from dance import logger
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmvae import scMVAE
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.transforms.preprocess import calculate_log_library_size
from dance.utils import set_seed


def parameter_setting():
    parser = argparse.ArgumentParser(description="Single cell Multi-omics data analysis")

    parser.add_argument("--workdir", "-wk", type=str, default="./new_test", help="work path")
    parser.add_argument("--outdir", "-od", type=str, default="./new_test", help="Output path")

    parser.add_argument("--lr", type=float, default=1E-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
    parser.add_argument("--eps", type=float, default=0.01, help="eps")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")

    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument('-seed', '--seed', type=int, default=1, help='Random seed for repeat results')
    parser.add_argument("--latent", "-l", type=int, default=10, help="latent layer dim")
    parser.add_argument("--max_epoch", "-me", type=int, default=25, help="Max epoches")
    parser.add_argument("--max_iteration", "-mi", type=int, default=3000, help="Max iteration")
    parser.add_argument("--anneal_epoch", "-ae", type=int, default=200, help="Anneal epoch")
    parser.add_argument("--epoch_per_test", "-ept", type=int, default=1,
                        help="Epoch per test, must smaller than max iteration.")
    parser.add_argument("--max_ARI", "-ma", type=int, default=-200, help="initial ARI")
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2")
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("--final_rate", type=float, default=1e-4)
    parser.add_argument("--scale_factor", type=float, default=4)

    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)

    return parser


if __name__ == "__main__":
    parser = parameter_setting()
    args = parser.parse_args()
    assert args.max_iteration > args.epoch_per_test
    device = torch.device(args.device)
    args.lr = 0.001
    args.anneal_epoch = 200
    res = None
    logger.info(f"\n{pprint.pformat(vars(args))}")
    file_root_path = Path(args.root_path, args.subtask).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)
        dataset = JointEmbeddingNIPSDataset(args.subtask, root="./data/joint_embedding")
        data = dataset.load_data()

        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(data.mod["test_sol"].obs["cell_type"])
        data.mod["mod1"].obsm["labels"] = labels

        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        (x_train, y_train), _ = data.get_train_data(return_type="torch")
        (x_test, y_test), labels = data.get_test_data(return_type="torch")

        lib_mean1, lib_var1 = calculate_log_library_size(np.concatenate([x_train.numpy(), x_test.numpy()]))
        lib_mean2, lib_var2 = calculate_log_library_size(np.concatenate([y_train.numpy(), y_test.numpy()]))
        lib_mean1 = torch.from_numpy(lib_mean1)
        lib_var1 = torch.from_numpy(lib_var1)
        lib_mean2 = torch.from_numpy(lib_mean2)
        lib_var2 = torch.from_numpy(lib_var2)

        Nfeature1 = x_train.shape[1]
        Nfeature2 = y_train.shape[1]
        train_size = len(data.get_split_idx("train"))
        train = data_utils.TensorDataset(x_train, lib_mean1[:train_size], lib_var1[:train_size], lib_mean2[:train_size],
                                         lib_var2[:train_size], y_train)

        valid = data_utils.TensorDataset(x_test, lib_mean1[train_size:], lib_var1[train_size:], lib_mean2[train_size:],
                                         lib_var2[train_size:], y_test)

        total = data_utils.TensorDataset(torch.cat([x_train, x_test]), torch.cat([y_train, y_test]))

        total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)

        x_test = torch.cat([x_train, x_test])
        y_test = torch.cat([y_train, y_test])
        labels = torch.from_numpy(le.fit_transform(data.mod["test_sol"].obs["cell_type"]))  #这里大概会有问题，很可能就是降维的问题
        model = scMVAE(
            encoder_1=[Nfeature1, 1024, 128, 128],
            hidden_1=128,
            Z_DIMS=22,
            decoder_share=[22, 128, 256],
            share_hidden=128,
            decoder_1=[128, 128, 1024],
            hidden_2=1024,
            encoder_l=[Nfeature1, 128],
            hidden3=128,
            encoder_2=[Nfeature2, 1024, 128, 128],
            hidden_4=128,
            encoder_l1=[Nfeature2, 128],
            hidden3_1=128,
            decoder_2=[128, 128, 1024],
            hidden_5=1024,
            drop_rate=0.1,
            log_variational=True,
            Type="ZINB",
            device=device,
            n_centroids=22,
            penality="GMM",
            model=1,
        )
        model.to(device)
        model.init_gmm_params(total_loader)
        model.fit(args, train, valid, args.final_rate, args.scale_factor, device)

        # embeds = model.predict(x_test, y_test).cpu().numpy()
        score = model.score(x_test, y_test, labels)
        score.update(model.score(x_test, y_test, labels, adata_sol=data.data['test_sol'], metric="openproblems"))
        score["ARI"] = score["dance_ari"]
        del score["dance_ari"]
        wandb.log(score)
        wandb.finish()
        torch.cuda.empty_cache()
        # score.update({
        #     'seed': args.seed + k,
        #     'subtask': args.subtask,
        #     'method': 'scmvae',
        # })

        # if res is not None:
        #     res = res.append(score, ignore_index=True)
        # else:
        #     for s in score:
        #         score[s] = [score[s]]
        #     res = pd.DataFrame(score)

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path,
                       required_funs=["AlignMod", "FilterCellsCommonMod", "FilterCellsCommonMod",
                                      "SetConfig"], required_indexes=[2, 11, 14, sys.maxsize], metric="ARI")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce scMVAE on other samples, please refer to command lines belows:

GEX-ADT:
$ python scmvae.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
$ python scmvae.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
