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
from dance.modules.multi_modality.joint_embedding.jae import JAEWrapper
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--subtask", default="openproblems_bmmc_cite_phase2",
        choices=["GSE140203_BRAIN_atac2gex", "openproblems_bmmc_cite_phase2", "openproblems_bmmc_multiome_phase2"])
    parser.add_argument("-d", "--data_folder", default="./data/joint_embedding")
    parser.add_argument("-pre", "--pretrained_folder", default="./data/joint_embedding/pretrained")
    parser.add_argument("-csv", "--csv_path", default="decoupled_lsi.csv")
    parser.add_argument("-seed", "--seed", default=1, type=int)
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
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
        dataset = JointEmbeddingNIPSDataset(args.subtask, root=args.data_folder)
        data = dataset.load_data()
        # Prepare preprocessing pipeline and apply it to data
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)
        logger.warning(data)
        (X_mod1_train, X_mod2_train), (cell_type, batch_label, phase_label, S_score,
                                       G2M_score) = data.get_train_data(return_type="torch")
        (X_mod1_test, X_mod2_test), (cell_type_test, _, _, _, _) = data.get_test_data(return_type="torch")
        X_train = torch.cat([X_mod1_train, X_mod2_train], dim=1)
        phase_score = torch.cat([S_score[:, None], G2M_score[:, None]], 1)
        X_test = torch.cat([X_mod1_test, X_mod2_test], dim=1)
        X_test = torch.cat([X_train, X_test]).float().to(device)
        test_id = np.arange(X_test.shape[0])
        labels = torch.cat([cell_type, cell_type_test]).numpy()
        adata_sol = data.data['test_sol']  # [data._split_idx_dict['test']]

        model = JAEWrapper(args, num_celL_types=int(cell_type.max() + 1), num_batches=int(batch_label.max() + 1),
                           num_phases=phase_score.shape[1], num_features=X_train.shape[1])
        model.fit(X_train, cell_type, batch_label, phase_score, max_epochs=50)

        embeds = model.predict(X_test, test_id).cpu().numpy()
        print(embeds)

        score = model.score(X_test, test_id, labels, metric="clustering")
        score.update(model.score(X_test, test_id, labels, adata_sol=adata_sol, metric="openproblems"))
        score.update({
            'subtask': args.subtask,
            'method': 'jae',
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
"""To reproduce JAE on other samples, please refer to command lines belows:

GEX-ADT:
$ python jae.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
$ python jae.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
