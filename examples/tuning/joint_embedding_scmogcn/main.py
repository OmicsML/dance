import argparse
import os
import pprint
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import wandb
from dance import logger
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmogcn import ScMoGCNWrapper
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.transforms.graph.cell_feature_graph import CellFeatureBipartiteGraph
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2",
                        choices=["openproblems_bmmc_cite_phase2", "openproblems_bmmc_multiome_phase2"])
    parser.add_argument("-d", "--data_folder", default="./data/joint_embedding")
    parser.add_argument("-pre", "--pretrained_folder", default="./data/joint_embedding/pretrained")
    parser.add_argument("-csv", "--csv_path", default="decoupled_lsi.csv")
    parser.add_argument("-l", "--layers", default=3, type=int, choices=[3, 4, 5, 6, 7])
    parser.add_argument("-dis", "--disable_propagation", default=0, type=int, choices=[0, 1, 2])
    parser.add_argument("-seed", "--seed", default=1, type=int)
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-bs", "--batch_size", default=512, type=int)
    parser.add_argument("-nm", "--normalize", default=1, type=int, choices=[0, 1])
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")

    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)

    args = parser.parse_args()

    device = args.device
    pre_normalize = bool(args.normalize)
    torch.set_num_threads(args.cpus)
    rndseed = args.seed
    set_seed(rndseed)

    res = None
    logger.info(f"\n{pprint.pformat(vars(args))}")
    file_root_path = Path(args.root_path, args.subtask).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)
        dataset = JointEmbeddingNIPSDataset(args.subtask, root=args.data_folder, preprocess="aux", normalize=True)
        data = dataset.load_data()
        train_size = len(data.get_split_idx("train"))

        data = CellFeatureBipartiteGraph(cell_feature_channel="X_pca", mod="mod1")(data)
        data = CellFeatureBipartiteGraph(cell_feature_channel="X_pca", mod="mod2")(data)
        # data.set_config(
        #     feature_mod=["mod1", "mod2"],
        #     label_mod=["mod1", "mod1", "mod1", "mod1", "mod1"],
        #     feature_channel=["X_pca", "X_pca"],
        #     label_channel=["cell_type", "batch_label", "phase_labels", "S_scores", "G2M_scores"],
        # )
        (x_mod1, x_mod2), (cell_type, batch_label, phase_label, S_score, G2M_score) = data.get_data(return_type="torch")
        phase_score = torch.cat([S_score[:, None], G2M_score[:, None]], 1)
        test_id = np.arange(x_mod1.shape[0])
        labels = cell_type.numpy()
        adata_sol = data.data['test_sol']  # [data._split_idx_dict['test']]
        model = ScMoGCNWrapper(args, num_celL_types=int(cell_type.max() + 1), num_batches=int(batch_label.max() + 1),
                               num_phases=phase_score.shape[1], num_features=x_mod1.shape[1] + x_mod2.shape[1])
        model.fit(
            g_mod1=data.data["mod1"].uns["g"],
            g_mod2=data.data["mod2"].uns["g"],
            train_size=train_size,
            cell_type=cell_type,
            batch_label=batch_label,
            phase_score=phase_score,
        )

        embeds = model.predict(test_id).cpu().numpy()
        score = model.score(test_id, labels, metric="clustering")
        score.update(model.score(test_id, labels, adata_sol=adata_sol, metric="openproblems"))
        score.update({
            'subtask': args.subtask,
            'method': 'scmogcn',
        })

        score["ARI"] = score["dance_ari"]
        del score["dance_ari"]
        wandb.log(score)
        wandb.finish()
        torch.cuda.empty_cache()

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
"""To reproduce scMoGCN on other samples, please refer to command lines belows:

GEX-ADT:
$ python scmogcn.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
$ python scmogcn.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
