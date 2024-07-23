import argparse
import gc
import os
import pprint
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import torch
import torch.utils.data as data_utils
from sklearn import preprocessing

import dance.utils.metrics as metrics
import wandb
from dance import logger
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.dcca import DCCA
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed


def parameter_setting():
    parser = argparse.ArgumentParser(description="Single cell Multi-omics data analysis")

    parser.add_argument("--latent_fusion", "-olf1", type=str, default="First_simulate_fusion.csv",
                        help="fusion latent code file")
    parser.add_argument("--latent_1", "-ol1", type=str, default="scRNA_latent_combine.csv",
                        help="first latent code file")
    parser.add_argument("--latent_2", "-ol2", type=str, default="scATAC_latent.csv", help="seconde latent code file")
    parser.add_argument("--denoised_1", "-od1", type=str, default="scRNA_seq_denoised.csv",
                        help="outfile for denoised file1")
    parser.add_argument("--normalized_1", "-on1", type=str, default="scRNA_seq_normalized_combine.tsv",
                        help="outfile for normalized file1")
    parser.add_argument("--denoised_2", "-od2", type=str, default="scATAC_seq_denoised.csv",
                        help="outfile for denoised file2")

    parser.add_argument("--workdir", "-wk", type=str, default="./new_test/", help="work path")
    parser.add_argument("--outdir", "-od", type=str, default="./new_test/", help="Output path")

    parser.add_argument("--lr", type=float, default=1E-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
    parser.add_argument("--eps", type=float, default=0.01, help="eps")

    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")

    parser.add_argument("--seed", type=int, default=1, help="Random seed for repeat results")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--latent", "-l", type=int, default=10, help="latent layer dim")
    parser.add_argument("--max_epoch", "-me", type=int, default=10, help="Max epoches")
    parser.add_argument("--max_iteration", "-mi", type=int, default=3000, help="Max iteration")
    parser.add_argument("--anneal_epoch", "-ae", type=int, default=200, help="Anneal epoch")
    parser.add_argument("--epoch_per_test", "-ept", type=int, default=5, help="Epoch per test")
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

    args.sf1 = 5
    args.sf2 = 1
    args.cluster1 = args.cluster2 = 4
    args.lr1 = 0.01
    args.flr1 = 0.001
    args.lr2 = 0.005
    args.flr2 = 0.0005

    res = None
    logger.info(f"\n{pprint.pformat(vars(args))}")
    file_root_path = Path(args.root_path, args.subtask).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)
        #         model = DCCA(layer_e_1=[Nfeature1, 128], hidden1_1=128, Zdim_1=4, layer_d_1=[4, 128], hidden2_1=128,
        #                      layer_e_2=[Nfeature2, 1500, 128], hidden1_2=128, Zdim_2=4, layer_d_2=[4], hidden2_2=4, args=args,
        #                      Type_1="NB", Type_2="Bernoulli", ground_truth1=torch.cat([train_labels, test_labels]), cycle=1,
        #                      attention_loss="Eucli")  # yapf: disable
        dataset = JointEmbeddingNIPSDataset(args.subtask, root="./data/joint_embedding")
        data = dataset.load_data()

        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(data.mod["test_sol"].obs["cell_type"])
        data.mod["mod2"].obsm["size_factors"] = np.sum(data.mod["mod2"].X.todense() if scipy.sparse.issparse(data.mod["mod2"].X) else data.mod["mod2"].X, 1) / 100
        # data.mod["mod1"].obsm["size_factors"] = data.mod["mod1"].obs["size_factors"]
        data.mod["mod1"].obsm["size_factors"] = np.sum(data.mod["mod1"].X.todense() if scipy.sparse.issparse(data.mod["mod1"].X) else data.mod["mod1"].X, 1) / 100
        data.mod["mod1"].obsm["labels"] = labels

        # data.set_config(feature_mod=["mod1", "mod2", "mod1", "mod2", "mod1", "mod2"], label_mod="mod1",
        #                 feature_channel_type=["layers", "layers", None, None, "obsm", "obsm"],
        #                 feature_channel=["counts", "counts", None, None, "size_factors",
        #                                 "size_factors"], label_channel="labels")
        (x_train, y_train, x_train_raw, y_train_raw, x_train_size,
        y_train_size), train_labels = data.get_train_data(return_type="torch")
        (x_test, y_test, x_test_raw, y_test_raw, x_test_size,
        y_test_size), test_labels = data.get_test_data(return_type="torch")

        Nfeature1 = x_train.shape[1]
        Nfeature2 = y_train.shape[1]

        device = torch.device(args.device)
        train = data_utils.TensorDataset(x_train.float(), x_train_raw, x_train_size.float(), y_train.float(), y_train_raw,
                                        y_train_size.float())

        train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

        test = data_utils.TensorDataset(x_test.float(), x_test_raw, x_test_size.float(), y_test.float(), y_test_raw,
                                        y_test_size.float())

        test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False)

        total = data_utils.TensorDataset(
            torch.cat([x_train, x_test]).float(), torch.cat([x_train_raw, x_test_raw]),
            torch.cat([x_train_size, x_test_size]).float(),
            torch.cat([y_train, y_test]).float(), torch.cat([y_train_raw, y_test_raw]),
            torch.cat([y_train_size, y_test_size]).float())

        total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)
        model = DCCA(layer_e_1=[Nfeature1, 128], hidden1_1=128, Zdim_1=50, layer_d_1=[50, 128], hidden2_1=128,
                     layer_e_2=[Nfeature2, 1500, 128], hidden1_2=128, Zdim_2=50, layer_d_2=[50], hidden2_2=50,
                     args=args, ground_truth1=torch.cat([train_labels, test_labels]), Type_1="NB", Type_2="Bernoulli",
                     cycle=1, attention_loss="Eucli").to(device)
        model.to(device)
        model.fit(train_loader, test_loader, total_loader, "RNA")

        emb1, emb2 = model.predict(total_loader)
        embeds = np.concatenate([emb1, emb2], 1)
        print(embeds)

        adata = ad.AnnData(
            X=embeds,
            obs=data.mod["mod1"].obs,
        )
        adata_sol = data.mod["test_sol"]
        adata = adata[adata_sol.obs_names]
        adata_sol.obsm['X_emb'] = adata.X
        score = metrics.labeled_clustering_evaluate(adata, adata_sol)
        score.update(metrics.integration_openproblems_evaluate(adata_sol))
        # score.update({
        #     'seed': args.seed + k,
        #     'subtask': args.subtask,
        #     'method': 'dcca',
        # })

        # if res is not None:
        #     res = res.append(score, ignore_index=True)
        # else:
        #     for s in score:
        #         score[s] = [score[s]]
        #     res = pd.DataFrame(score)
        score["ARI"]=score["dance_ari"]
        del score["dance_ari"]
        wandb.log(score)
        wandb.finish()
        torch.cuda.empty_cache()
        #主要是报错时没有执行这些命令导致的，我感觉
        del data,model,adata_sol,adata,embeds,emb1, emb2,total_loader,total,test_loader,test,train_loader,train,Nfeature2,Nfeature1
        del x_train, y_train, x_train_raw, y_train_raw, x_train_size,y_train_size,train_labels,x_test, y_test, x_test_raw, y_test_raw, x_test_size,y_test_size, test_labels
        del labels,le,dataset,score
        gc.collect()
    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path, required_funs=["AlignMod","FilterCellsCommonMod","FilterCellsCommonMod","SetConfig"],
                       required_indexes=[2,11,14,sys.maxsize], metric="ARI")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)

"""To reproduce DCCA on other samples, please refer to command lines belows:

GEX-ADT:
$ python dcca.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
$ python dcca.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
