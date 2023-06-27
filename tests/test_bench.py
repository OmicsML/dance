import gc
import logging
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

HOME_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = HOME_DIR / "examples"
logging.info(f"{HOME_DIR=}")

SKIP_LIST: List[str] = [
    "joint_embedding-dcca",  # OOM with 64GB mem and V100 GPU (succeed with 80GB mem)
    "joint_embedding-jae",  # long run (~1 hr using V100 GPU)
    "joint_embedding-scmogcn",  # long run (~1 hr using V100 GPU)
]

light_options_dict: Dict[str, Tuple[str, str]] = {
    # {task}-{method}-{dataset}: {command_line_options}
    # Single modality
    "cell_type_annotation-actinn-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --lambd 0.0001 --device cuda --num_epochs 2 --runs 1",
    "cell_type_annotation-celltypist-spleen": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203",
    "cell_type_annotation-scdeepsort-spleen": "--tissue Spleen --test_data 1759 --device cuda --n_epochs 2",
    "cell_type_annotation-singlecellnet-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759",
    "cell_type_annotation-svm-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759",
    "clustering-graphsc-10X_PBMC": "--dataset 10X_PBMC --epochs 2",
    "clustering-scdcc-10X_PBMC": "--dataset 10X_PBMC --label_cells_files label_10X_PBMC.txt --gamma 1.5 --epochs 2 --pretrain_epochs 2",
    "clustering-scdeepcluster-10X_PBMC": "--dataset 10X_PBMC --pretrain_epochs 2",
    "clustering-scdsc-10X_PBMC": "--dataset 10X_PBMC --method cosine --topk 30 --v 7 --binary_crossentropy_loss 0.75 --ce_loss 0.5 --re_loss 0.1 --zinb_loss 2.5 --sigma 0.4 --epochs 2 --pretrain_epochs 2",
    "clustering-sctag-10X_PBMC": "--pretrain_epochs 2 --epochs 2 --dataset 10X_PBMC --w_a 0.01 --w_x 3 --w_c 0.1 --dropout 0.5",
    "imputation-deepimpute-brain": "--train_dataset mouse_brain_data --filetype h5 --hidden_dim 200 --dropout 0.4 --n_epochs 2 --gpu 0",
    "imputation-graphsci-brain": "--train_dataset mouse_brain_data --gpu 0 --n_epochs 2",
    "imputation-scgnn-brain": "--train_dataset mouse_brain_data --Regu_epochs 2 --EM_epochs 2 --cluster_epochs 2 --GAEepochs 2 --gpu 0",
    # Multi modality
    "joint_embedding-dcca-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda --max_epoch 2 --max_iteration 10 --anneal_epoch 2 --epoch_per_test 2",
    "joint_embedding-jae-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda",
    "joint_embedding-scmogcn-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda",
    "joint_embedding-scmvae-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda --max_epoch 2 --anneal_epoch 2 --epoch_per_test 2 --max_iteration 10",
    "match_modality-cmae-gex2adt_subset": "--subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda --max_epochs 2",
    "match_modality-scmm-gex2adt_subset": "--subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda --epochs 2",
    "match_modality-scmogcn-gex2adt_subset": "--subtask openproblems_bmmc_cite_phase2_rna_subset --threshold_quantile 0.85 --device cuda --epochs 2",
    "predict_modality-babel-gex2adt_subset": "--subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda --max_epochs 2 --earlystop 2",
    "predict_modality-cmae-gex2adt_subset": "--subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda --max_epochs 2",
    "predict_modality-scmm-gex2adt_subset": "--subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda --epochs 2",
    "predict_modality-scmogcn-gex2adt_subset": "--subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda --epoch 2",
    # Spatial
    "cell_type_deconvo-card-card_synth": "--dataset CARD_synthetic --max_iter 2",
    "cell_type_deconvo-dstg-spotlight_synth": "--dataset SPOTLight_synthetic --nhid 32 --lr .1 --epochs 25 --device cuda",
    "cell_type_deconvo-spatialdecon-card_synth": "--dataset CARD_synthetic --lr .01 --max_iter 2 --bias 1 --device cuda",
    "cell_type_deconvo-spotlight-card_synth": "--dataset CARD_synthetic --lr .1 --max_iter 2 --rank 8 --bias 0 --device cuda",
    "spatial_domain-louvain-151507": "--sample_number 151507 --seed 10",
    "spatial_domain-spagcn-151507": "--sample_number 151507 --lr 0.009",
    "spatial_domain-stagate-151507": "--sample_number 151507 --seed 2021",
    "spatial_domain-stlearn-151507": "--n_clusters 20 --sample_number 151507 --seed 0",
}  # yapf: disable

full_options_dict: Dict[str, Tuple[str, str]] = {
    # Single modality
    "cell_type_annotation-actinn-brain": "--species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695 --lambd 0.1 --device cuda:0",
    "cell_type_annotation-actinn-kidney": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203 --lambd 0.01 --device cuda:0",
    "cell_type_annotation-actinn-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --lambd 0.0001 --device cuda:0",
    "cell_type_annotation-celltypist-brain": "--species mouse --tissue Brain --train_dataset 753 --test_dataset 2695",
    "cell_type_annotation-celltypist-kidney": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759",
    "cell_type_annotation-celltypist-spleen": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203",
    "cell_type_annotation-scdeepsort-brain": "--tissue Brain --test_data 2695 --device cuda",
    "cell_type_annotation-scdeepsort-kidney": "--tissue Kidney --test_data 203 --device cuda",
    "cell_type_annotation-scdeepsort-spleen": "--tissue Spleen --test_data 1759 --device cuda",
    "cell_type_annotation-singlecellnet-brain": "--species mouse --tissue Brain --train_dataset 753 --test_dataset 2695",
    "cell_type_annotation-singlecellnet-kidney": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203",
    "cell_type_annotation-singlecellnet-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759",
    "cell_type_annotation-svm-brain": "--species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695",
    "cell_type_annotation-svm-kidney": "--species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203",
    "cell_type_annotation-svm-spleen": "--species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759",
    "clustering-graphsc-10X_PBMC": "--dataset 10X_PBMC",
    "clustering-graphsc-mouse_ES_cell": "--dataset mouse_ES_cell",
    "clustering-graphsc-mouse_bladder_cell": "--dataset mouse_bladder_cell",
    "clustering-graphsc-worm_neuron_cell": "--dataset worm_neuron_cell",
    "clustering-scdcc-10X_PBMC": "--dataset 10X_PBMC --label_cells_files label_10X_PBMC.txt --gamma 1.5",
    "clustering-scdcc-mouse_ES_cell": "--dataset mouse_ES_cell --label_cells_files label_mouse_ES_cell.txt --gamma 1 --ml_weight 0.8 --cl_weight 0.8",
    "clustering-scdcc-mouse_bladder_cell": "--dataset mouse_bladder_cell --label_cells_files label_mouse_bladder_cell.txt --gamma 1.5 --pretrain_epochs 100 --sigma 3",
    "clustering-scdcc-worm_neuron_cell": "--dataset worm_neuron_cell --label_cells_files label_worm_neuron_cell.txt --gamma 1 --pretrain_epochs 300",
    "clustering-scdeepcluster-10X_PBMC": "--dataset 10X_PBMC",
    "clustering-scdeepcluster-mouse_ES_cell": "--dataset mouse_ES_cell",
    "clustering-scdeepcluster-mouse_bladder_cell": "--dataset mouse_bladder_cell --pretrain_epochs 300 --sigma 2.75",
    "clustering-scdeepcluster-worm_neuron_cell": "--dataset worm_neuron_cell --pretrain_epochs 300",
    "clustering-scdsc-10X_PBMC": "--dataset 10X_PBMC --method cosine --topk 30 --v 7 --binary_crossentropy_loss 0.75 --ce_loss 0.5 --re_loss 0.1 --zinb_loss 2.5 --sigma 0.4",
    "clustering-scdsc-mouse_ES_cell": "--dataset mouse_ES_cell --method cosine --topk 50 --v 7 --binary_crossentropy_loss 0.1 --ce_loss 0.01 --re_loss 1.5 --zinb_loss 0.5 --sigma 0.1",
    "clustering-scdsc-mouse_bladder_cell": "--dataset mouse_bladder_cell --method correlation --topk 50 --v 7 --binary_crossentropy_loss 2.5 --ce_loss 0.1 --re_loss 0.5 --zinb_loss 1.5 --sigma 0.6",
    "clustering-scdsc-worm_neuron_cell": "--dataset worm_neuron_cell --method correlation --topk 20 --v 7 --binary_crossentropy_loss 2 --ce_loss 2 --re_loss 3 --zinb_loss 0.1 --sigma 0.4",
    "clustering-sctag-10X_PBMC": "--pretrain_epochs 100 --dataset 10X_PBMC --w_a 0.01 --w_x 3 --w_c 0.1 --dropout 0.5",
    "clustering-sctag-mouse_ES_cell": "--dataset mouse_ES_cell --w_a 0.01 --w_x 2 --w_c 0.25 --k 1",
    "clustering-sctag-mouse_bladder_cell": "--pretrain_epochs 100 --dataset mouse_bladder_cell --w_a 0.01 --w_x 0.75 --w_c 1",
    "clustering-sctag-worm_neuron_cell": "--pretrain_epochs 100 --dataset worm_neuron_cell --w_a 0.1 --w_x 2.5 --w_c 3",
    "imputation-deepimpute-brain": "--train_dataset mouse_brain_data --filetype h5 --hidden_dim 200 --dropout 0.4",
    "imputation-deepimpute-embryo": "--train_dataset mouse_embryo_data --filetype gz --hidden_dim 200 --dropout 0.4",
    "imputation-graphsci-brain": "--train_dataset mouse_brain_data --gpu 0",
    "imputation-graphsci-embryo": "--train_dataset mouse_embryo_data --gpu 0",
    "imputation-scgnn-brain": "--train_dataset mouse_brain_data --gpu 0",
    "imputation-scgnn-embryo": "--train_dataset mouse_embryo_data --gpu 0",
    # Multi modality
    "joint_embedding-dcca-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda",
    "joint_embedding-dcca-gex_atac": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    "joint_embedding-jae-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda",
    "joint_embedding-jae-gex_atac": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    "joint_embedding-scmogcn-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda",
    "joint_embedding-scmogcn-gex_atac": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    "joint_embedding-scmvae-gex_adt": "--subtask openproblems_bmmc_cite_phase2 --device cuda",
    "joint_embedding-scmvae-gex_atac": "--subtask openproblems_bmmc_multiome_phase2 --device cuda",
    "match_modality-cmae-gex_adt": "--subtask openproblems_bmmc_cite_phase2_rna --device cuda",
    "match_modality-cmae-gex_atac": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "match_modality-scmm-gex_adt": "--subtask openproblems_bmmc_cite_phase2_rna --device cuda",
    "match_modality-scmm-gex_atac": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "match_modality-scmogcn-gex_adt": "--subtask openproblems_bmmc_cite_phase2_rna --device cuda",
    "match_modality-scmogcn-gex_atac": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "predict_modality-babel-adt2gex": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-babel-atac2gex": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-babel-gex2adt": "--subtask openproblems_bmmc_cite_phase2_rna --device cuda",
    "predict_modality-babel-gex2atac": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "predict_modality-cmae-adt2gex": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-cmae-atac2gex": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-cmae-gex2adt": "--subtask openproblems_bmmc_cite_phase2_rna --device cuda",
    "predict_modality-cmae-gex2atac": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "predict_modality-scmm-adt2gex": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-scmm-atac2gex": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-scmm-gex2adt": "--subtask openproblems_bmmc_cite_phase2_rna --device cuda",
    "predict_modality-scmm-gex2atac": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    "predict_modality-scmogcn-adt2gex": "--subtask openproblems_bmmc_cite_phase2_mod2 --device cuda",
    "predict_modality-scmogcn-atac2gex": "--subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda",
    "predict_modality-scmogcn-gex2adt": "--subtask oopenproblems_bmmc_cite_phase2_rna --device cuda",
    "predict_modality-scmogcn-gex2atac": "--subtask openproblems_bmmc_multiome_phase2_rna --device cuda",
    # Spatial
    "cell_type_deconvo-card-card_synth": "--dataset CARD_synthetic",
    "cell_type_deconvo-card-gse174746": "--dataset GSE174746 --location_free",
    "cell_type_deconvo-card-spotlight_synth": "--dataset SPOTLight_synthetic --location_free",
    "cell_type_deconvo-dstg-card_synth": "--dataset CARD_synthetic --nhid 16 --lr .001",
    "cell_type_deconvo-dstg-gse174746": "--dataset GSE174746 --nhid 16 --lr .0001",
    "cell_type_deconvo-dstg-spotlight_synth": "--dataset SPOTLight_synthetic --nhid 32 --lr .1 --epochs 25",
    "cell_type_deconvo-spatialdecon-card_synth": "--dataset CARD_synthetic --lr .01 --max_iter 2250 --bias 1",
    "cell_type_deconvo-spatialdecon-gse174746": "--dataset GSE174746 --lr .0001 --max_iter 20000 --bias 1",
    "cell_type_deconvo-spatialdecon-spotlight_synth": "--dataset SPOTLight_synthetic --lr .01 --max_iter 500 --bias 1",
    "cell_type_deconvo-spotlight-card_synth": "--dataset CARD_synthetic --lr .1 --max_iter 100 --rank 8 --bias 0",
    "cell_type_deconvo-spotlight-gse174746": "--dataset GSE174746 --lr .1 --max_iter 15000 --rank 4 --bias 0",
    "cell_type_deconvo-spotlight-spotlight_synth": "--dataset SPOTLight_synthetic --lr .1 --max_iter 150 --rank 10 --bias 0",
    "spatial_domain-louvain-151507": "--sample_number 151507 --seed 10",
    "spatial_domain-louvain-151673": "--sample_number 151673 --seed 5",
    "spatial_domain-louvain-151676": "--sample_number 151676 --seed 203",
    "spatial_domain-spagcn-151507": "--sample_number 151507 --lr 0.009",
    "spatial_domain-spagcn-151673": "--sample_number 151673 --lr 0.1",
    "spatial_domain-spagcn-151676": "--sample_number 151676 --lr 0.02",
    "spatial_domain-stagate-151507": "--sample_number 151507 --seed 2021",
    "spatial_domain-stagate-151673": "--sample_number 151673 --seed 16",
    "spatial_domain-stagate-151676": "--sample_number 151676 --seed 2030",
    "spatial_domain-stlearn-151507": "--n_clusters 20 --sample_number 151507 --seed 0",
    "spatial_domain-stlearn-151673": "--n_clusters 20 --sample_number 151673 --seed 93",
    "spatial_domain-stlearn-151676": "--n_clusters 20 --sample_number 151676 --seed 11",
}  # yapf: disable


def find_script_path(script_name: str, task_name: str) -> Path:
    for dir_, subdirs, files in os.walk(SCRIPTS_DIR):
        if script_name in files and task_name in dir_:
            logging.info(f"Found {script_name} under {dir_}")
            return Path(dir_)
    raise FileNotFoundError(f"Failed to locate {script_name!r} for task {task_name!r} under {SCRIPTS_DIR!s}")


def run_benchmarks(name, options_dict):
    # Check to see if the test run name is contained in any element of the
    # SKIP_LIST and return immediatel if so
    if any(map(lambda to_skip: to_skip in name, SKIP_LIST)):
        logging.warning(f"Skipping run {name!r} as it is contained in one of the elements in {SKIP_LIST=}")
        return

    task_name = name.split("-")[0]
    script_name = name.split("-")[1] + ".py"
    os.chdir(find_script_path(script_name, task_name))

    # Overwrite sysargv and run test script
    args = options_dict[name]
    sys.argv = [None] + (args.split(" ") if args else [])
    logging.info(f"Start running [{name}] with {args=!r}")
    t = time.perf_counter()
    runpy.run_path(script_name, run_name="__main__")
    t = time.perf_counter() - t
    logging.info(f"Finished running [{name}] - took {int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}")

    # Post run cleanup
    gc.collect()


@pytest.mark.parametrize("name", sorted(full_options_dict))
@pytest.mark.full_test
def test_bench_full(name):
    run_benchmarks(name, full_options_dict)


@pytest.mark.parametrize("name", sorted(light_options_dict))
@pytest.mark.light_test
def test_bench_light(name):
    run_benchmarks(name, light_options_dict)
