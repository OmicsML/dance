import argparse
import gc
import os
import pprint
import re
import sys
from collections import Counter
from pathlib import Path

# Keep preprocessing caches out of the repository and enable PyTorch's
# fragmentation-resistant CUDA allocator for every newly isolated run.
_cache_root = Path(os.environ.get("NO_CELL_QC_CACHE_DIR", "/tmp/dance_scmogcn_no_cell_qc"))
for _env_name, _subdir in (
    ("NUMBA_CACHE_DIR", "numba"),
    ("MPLCONFIGDIR", "matplotlib"),
    ("XDG_CACHE_HOME", "xdg"),
):
    os.environ.setdefault(_env_name, str(_cache_root / _subdir))
    Path(os.environ[_env_name]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from prefix_cache import PipelinePrefixCache, canonical_prefix

import wandb
from dance import logger
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmogcn import ScMoGCNWrapper
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.transforms.graph.cell_feature_graph import CellFeatureBipartiteGraph
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--subtask", default="openproblems_bmmc_cite_phase2", choices=[
            "GSE140203_BRAIN_atac2gex", "openproblems_bmmc_cite_phase2", "openproblems_bmmc_multiome_phase2",
            "GSE140203_SKIN_atac2gex", "openproblems_2022_multi_atac2gex"
        ])
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
    parser.add_argument("--preprocess", type=str, default=None)

    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--run_step3", action="store_true",
                        help="Explicitly run step-3 parameter sweeps after the pipeline sweep.")
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument(
        "--prefix_cache_root", default=None, type=str,
        help="NFS directory for exact deterministic preprocessing-prefix caches. Defaults below --data_folder.")
    parser.add_argument("--prefix_cache_depth", default=5, type=int,
                        help="Number of leading pipeline actions to cache (5 means indexes 0 through 4).")
    parser.add_argument("--disable_prefix_cache", action="store_true")

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
    prefix_cache_root = Path(args.prefix_cache_root or Path(args.data_folder, ".dance_prefix_cache", "scmogcn"))
    prefix_counts = Counter()
    if not args.disable_prefix_cache and "run_kwargs" in pipeline_planer.config:
        grouped_candidates = {}
        for candidate in pipeline_planer.config.run_kwargs:
            candidate_dict = dict(candidate)
            prefix_token = tuple(
                sorted((key, str(value)) for key, value in candidate_dict.items() if (
                    match := re.match(r"pipeline\.(\d+)\.", key)) and int(match.group(1)) < args.prefix_cache_depth))
            if prefix_token not in grouped_candidates:
                grouped_candidates[prefix_token] = [candidate_dict, 0]
            grouped_candidates[prefix_token][1] += 1
        for candidate, count in grouped_candidates.values():
            candidate_pipeline = pipeline_planer.generate(**{args.tune_mode: candidate})
            prefix_counts[canonical_prefix(candidate_pipeline, args.prefix_cache_depth)] += count
        logger.info(f"Prefix cache enabled: root={prefix_cache_root}, depth={args.prefix_cache_depth}, "
                    f"reused_prefixes={sum(count > 1 for count in prefix_counts.values())}")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "True"

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)
        wandb_config = wandb.config
        if "run_kwargs" in pipeline_planer.config:
            if any(d == dict(wandb.config["run_kwargs"]) for d in pipeline_planer.config.run_kwargs):
                wandb_config = wandb_config["run_kwargs"]
            else:
                wandb.log({"skip": 1})
                wandb.finish()
                return
        try:
            dataset = JointEmbeddingNIPSDataset(args.subtask, root=args.data_folder, preprocess=args.preprocess)
            # Prepare preprocessing pipeline and apply it to data
            kwargs = {tune_mode: dict(wandb_config)}
            preprocessing_pipeline = pipeline_planer.generate(**kwargs)
            print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
            prefix = canonical_prefix(preprocessing_pipeline, args.prefix_cache_depth)

            def load_and_apply_prefix():
                prefix_data = dataset.load_data()
                for action_index in range(min(args.prefix_cache_depth, len(preprocessing_pipeline))):
                    preprocessing_pipeline[action_index](prefix_data)
                return prefix_data

            use_prefix_cache = (not args.disable_prefix_cache and args.prefix_cache_depth > 0
                                and prefix_counts.get(prefix, 0) > 1)
            if use_prefix_cache:
                prefix_cache = PipelinePrefixCache(prefix_cache_root, dataset, preprocessing_pipeline,
                                                   args.prefix_cache_depth, args.seed)
                data, prefix_cache_hit = prefix_cache.load_or_build(load_and_apply_prefix)
                logger.info(f"PREFIX_CACHE_RESULT hit={prefix_cache_hit} reuse_count={prefix_counts[prefix]}")
                pipeline_start = args.prefix_cache_depth
            else:
                data = dataset.load_data()
                pipeline_start = 0
                logger.info(f"PREFIX_CACHE_BYPASS reuse_count={prefix_counts.get(prefix, 0)}")

            for action_index in range(pipeline_start, len(preprocessing_pipeline)):
                preprocessing_pipeline[action_index](data)
            # train_idx=list(set(data.mod["meta1"].obs_names) & set(data.mod["mod1"].obs_names))
            train_name = [item for item in data.mod["mod1"].obs_names if item in data.mod["meta1"].obs_names]
            train_idx = [data.mod["mod1"].obs_names.get_loc(name) for name in train_name]
            test_idx = list({i for i in range(data.mod["mod1"].shape[0])}.difference(set(train_idx)))

            # train_size=data.mod["meta1"].shape[0]
            # test_size=data.mod["mod1"].shape[0]-train_size
            data.set_split_idx("train", train_idx)
            data.set_split_idx("test", test_idx)
            if args.preprocess != "aux":
                cell_type_labels = data.data['test_sol'].obs["cell_type"].to_numpy()
                cell_type_labels_unique = list(np.unique(cell_type_labels))
                c_labels = np.array([cell_type_labels_unique.index(item) for item in cell_type_labels])
                data.data['mod1'].obsm["cell_type"] = c_labels
                data.data["mod1"].obsm["S_scores"] = np.zeros(data.data['mod1'].shape[0])
                data.data["mod1"].obsm["G2M_scores"] = np.zeros(data.data['mod1'].shape[0])
                data.data["mod1"].obsm["batch_label"] = np.zeros(data.data['mod1'].shape[0])
                data.data["mod1"].obsm["phase_labels"] = np.zeros(data.data['mod1'].shape[0])

            # train_size = len(data.get_split_idx("train"))
            # In theory, meta1 should include all content from the first half of mod1, the order might have been shuffled during processing
            data = CellFeatureBipartiteGraph(cell_feature_channel="feature.cell", mod="mod1")(data)
            data = CellFeatureBipartiteGraph(cell_feature_channel="feature.cell", mod="mod2")(data)
            # data.set_config(
            #     feature_mod=["mod1", "mod2"],
            #     label_mod=["mod1", "mod1", "mod1", "mod1", "mod1"],
            #     feature_channel=["X_pca", "X_pca"],
            #     label_channel=["cell_type", "batch_label", "phase_labels", "S_scores", "G2M_scores"],
            # )
            (x_mod1, x_mod2), (cell_type, batch_label, phase_label, S_score,
                               G2M_score) = data.get_data(return_type="torch")
            phase_score = torch.cat([S_score[:, None], G2M_score[:, None]], 1)
            test_id = np.arange(x_mod1.shape[0])
            labels = cell_type.numpy()
            adata_sol = data.data['test_sol']  # [data._split_idx_dict['test']]
            model = ScMoGCNWrapper(args, num_celL_types=int(cell_type.max() + 1),
                                   num_batches=int(batch_label.max() + 1), num_phases=phase_score.shape[1],
                                   num_features=x_mod1.shape[1] + x_mod2.shape[1])
            model.fit(
                g_mod1=data.data["mod1"].uns["g"],
                g_mod2=data.data["mod2"].uns["g"],
                train_size=train_idx,
                cell_type=cell_type,
                batch_label=batch_label,
                phase_score=phase_score,
            )

            embeds = model.predict(test_id).cpu().numpy()
            score = model.score(test_id, labels, metric="clustering")
            # score.update(model.score(test_id, labels, adata_sol=adata_sol, metric="openproblems"))
            score.update({
                'subtask': args.subtask,
                'method': 'scmogcn',
            })

            score["ARI"] = score["dance_ari"]
            del score["dance_ari"]
            wandb.log(score)
            wandb.finish()
        finally:
            # del data,model,adata_sol,adata,embeds,emb1, emb2,total_loader,total,test_loader,test,train_loader,train,Nfeature2,Nfeature1
            # del x_train, y_train, x_train_raw, y_train_raw, x_train_size,y_train_size,train_labels,x_test, y_test, x_test_raw, y_test_raw, x_test_size,y_test_size, test_labels
            # del labels,le,dataset,score
            # variables_to_delete=["data","model","adata_sol","adata","embeds","emb1", "emb2","total_loader","total,test_loader","test,train_loader","train","Nfeature2","Nfeature1","x_train", "y_train", "x_train_raw", "y_train_raw", "x_train_size","y_train_size","train_labels","x_test", "y_test"," x_test_raw", y_test_raw, x_test_size,y_test_size, test_labels,labels,le,dataset,score]
            locals_keys = list(locals().keys())
            for var in locals_keys:
                try:
                    exec(f"del {var}")
                    logger.info(f"Deleted '{var}'")
                except NameError:
                    logger.info(f"Variable '{var}' does not exist, continuing...")
            torch.cuda.empty_cache()
            gc.collect()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.run_step3 and (args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params"):
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
