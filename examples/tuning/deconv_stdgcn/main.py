import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.stdgcn import stdGCNWrapper
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.utils import set_seed


def parse_arguments():
    parser = argparse.ArgumentParser(description="STdGCN Configuration Script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- wandb ---
    # These were not in a group, but seem fundamental
    wandb_group = parser.add_argument_group("Wandb", "params for wandb")
    wandb_group.add_argument("--seed", type=int, default=42)
    wandb_group.add_argument("--tune_mode", default="pipeline_params",
                             choices=["pipeline", "params", "pipeline_params"])
    wandb_group.add_argument("--count", type=int, default=1000)
    wandb_group.add_argument("--sweep_id", type=str, default=None)
    wandb_group.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    wandb_group.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)

    # --- paths ---
    # These were not in a group, but seem fundamental
    path_group = parser.add_argument_group("Paths", "Dataset and data directory paths.")
    path_group.add_argument('--dataset', type=str, default="CARD_synthetic",
                            help="Name of the dataset (e.g., CARD_synthetic).")
    path_group.add_argument('--datadir', type=str, default="data/spatial", help="Directory where the data is located.")

    # --- find_marker_genes_paras ---
    fmg_doc = """This module is used to preprocess the input data and identify marker genes [optional]."""
    fmg_group = parser.add_argument_group('Find Marker Genes Parameters', fmg_doc)

    fmg_group.add_argument(
        '--fmg_preprocess', dest='fmg_preprocess', action='store_true', default=True,
        help="Select whether the input expression data needs to be preprocessed for marker gene finding. "
        "If set, preprocessing occurs. Use --no-fmg-preprocess to disable.")
    fmg_group.add_argument('--no-fmg-preprocess', dest='fmg_preprocess', action='store_false',
                           help="Disable preprocessing for marker gene finding.")
    fmg_group.add_argument(
        '--fmg_normalize', dest='fmg_normalize', action='store_true', default=True,
        help="When 'fmg_preprocess'=True, normalize each cell/spot by total counts = 10,000. "
        "Use --no-fmg-normalize to disable.")
    fmg_group.add_argument('--no-fmg-normalize', dest='fmg_normalize', action='store_false',
                           help="Disable normalization in marker gene finding preprocessing.")
    fmg_group.add_argument(
        '--fmg_log', dest='fmg_log', action='store_true', default=True,
        help="When 'fmg_preprocess'=True, logarithmize (X=log(X+1)) the expression matrix. "
        "Use --no-fmg-log to disable.")
    fmg_group.add_argument('--no-fmg-log', dest='fmg_log', action='store_false',
                           help="Disable logarithmization in marker gene finding preprocessing.")
    fmg_group.add_argument(
        '--fmg_highly_variable_genes', dest='fmg_highly_variable_genes', action='store_true', default=False,
        help="When 'fmg_preprocess'=True, filter for highly variable genes. "
        "Disabled by default. Use --fmg-highly-variable-genes to enable.")
    fmg_group.add_argument(
        '--fmg_highly_variable_gene_num', type=int, default=None,
        help="When 'fmg_preprocess'=True and 'fmg_highly_variable_genes'=True, "
        "select the number of highly-variable genes to keep (e.g., 2000). Default: None (all HVGs based on criteria).")
    fmg_group.add_argument(
        '--fmg_regress_out', dest='fmg_regress_out', action='store_true', default=False,
        help="When 'fmg_preprocess'=True, regress out mitochondrial genes. "
        "Disabled by default. Use --fmg-regress-out to enable.")
    # 'scale' for fmg is implicitly True if preprocess is True, not listed as a separate sub-param in dict
    # Assuming 'scale' here means scale for PCA if preprocess is True.
    # The doc says "scaling data" as part of preprocess. The dict has 'scale': True, but it's not in find_marker_genes_paras.
    # I will add a general scale parameter for the fmg preprocessing block
    fmg_group.add_argument(
        '--fmg_scale',
        dest='fmg_scale',
        action='store_true',
        default=True,  # Assuming default based on doc
        help="When 'fmg_preprocess'=True, scale each gene to unit variance and zero mean. Use --no-fmg-scale to disable."
    )
    fmg_group.add_argument('--no-fmg-scale', dest='fmg_scale', action='store_false',
                           help="Disable scaling in marker gene finding preprocessing.")
    fmg_group.add_argument('--fmg_pca_components', type=int, default=30, dest='fmg_pca_components',
                           help="Number of principal components for PCA during marker gene finding.")
    fmg_group.add_argument('--marker_gene_method', type=str, default='logreg', choices=['logreg', 'wilcoxon'],
                           help="Method to identify cell type marker genes ('logreg' or 'wilcoxon').")
    fmg_group.add_argument('--top_gene_per_type', type=int, default=100,
                           help="Number of top marker genes per cell type to use.")
    fmg_group.add_argument(
        '--fmg_filter_wilcoxon_marker_genes', dest='fmg_filter_wilcoxon_marker_genes', action='store_true',
        default=True, help="When 'marker_gene_method'='wilcoxon', perform additional gene filtering. "
        "Use --no-fmg-filter-wilcoxon-marker-genes to disable.")
    fmg_group.add_argument('--no-fmg-filter-wilcoxon-marker-genes', dest='fmg_filter_wilcoxon_marker_genes',
                           action='store_false', help="Disable additional filtering for Wilcoxon marker genes.")
    fmg_group.add_argument(
        '--pvals_adj_threshold', type=float, default=0.10,
        help="When 'marker_gene_method'='wilcoxon' and filtering is enabled, "
        "corrected p-value threshold for genes.")
    fmg_group.add_argument(
        '--log_fold_change_threshold', type=float, default=1.0,
        help="When 'marker_gene_method'='wilcoxon' and filtering is enabled, "
        "log fold change threshold for genes.")
    fmg_group.add_argument(
        '--min_within_group_fraction_threshold', type=float, default=None,
        help="When 'marker_gene_method'='wilcoxon' and filtering is enabled, "
        "minimum fraction of expression within the cell type.")
    fmg_group.add_argument(
        '--max_between_group_fraction_threshold', type=float, default=None,
        help="When 'marker_gene_method'='wilcoxon' and filtering is enabled, "
        "maximum fraction of expression in the rest of cell types.")

    # --- pseudo_spot_simulation_paras ---
    pss_doc = """This module is used to simulate pseudo-spots."""
    pss_group = parser.add_argument_group('Pseudo-Spot Simulation Parameters', pss_doc)
    pss_group.add_argument(
        '--spot_num',
        type=int,
        default=100,  # Changed from 30000 for testing as per TODO
        help="The number of pseudo-spots to simulate.")
    pss_group.add_argument('--min_cell_num_in_spot', type=int, default=8,
                           help="Minimum number of cells in a pseudo-spot.")
    pss_group.add_argument('--max_cell_num_in_spot', type=int, default=12,
                           help="Maximum number of cells in a pseudo-spot.")
    pss_group.add_argument('--generation_method', type=str, default='celltype', choices=['cell', 'celltype'],
                           help="Pseudo-spot simulation method ('cell' or 'celltype').")
    pss_group.add_argument('--max_cell_types_in_spot', type=int, default=4,
                           help="When 'generation_method'='celltype', maximum number of cell types in a pseudo-spot.")

    # --- data_normalization_paras ---
    dn_doc = """This module is used for real- and pseudo- spots normalization."""
    dn_group = parser.add_argument_group('Data Normalization Parameters (Real/Pseudo Spots)', dn_doc)
    dn_group.add_argument(
        '--dn_normalize', dest='dn_normalize', action='store_true', default=True,
        help="Normalize real/pseudo spots by total counts = 10,000. Use --no-dn-normalize to disable.")
    dn_group.add_argument('--no-dn-normalize', dest='dn_normalize', action='store_false',
                          help="Disable normalization for real/pseudo spots.")
    dn_group.add_argument(
        '--dn_log', dest='dn_log', action='store_true', default=True,
        help="Logarithmize (X=log(X+1)) real/pseudo spot expression matrix. Use --no-dn-log to disable.")
    dn_group.add_argument('--no-dn-log', dest='dn_log', action='store_false',
                          help="Disable logarithmization for real/pseudo spots.")
    dn_group.add_argument(
        '--dn_scale', dest='dn_scale', action='store_true', default=False,
        help="Scale each gene to unit variance and zero mean for real/pseudo spots. Disabled by default.")

    # --- integration_for_adj_paras ---
    ifa_doc = """This module is used to integrate the normalized real- and pseudo- spots together to
construct the real-to-pseudo-spot link graph."""
    ifa_group = parser.add_argument_group('Integration for Adjacency Graph Parameters', ifa_doc)
    ifa_group.add_argument(
        '--adj_batch_removal_method', type=str, default=None, choices=['mnn', 'scanorama', 'combat', 'None'], help=
        "Batch removal method for adjacency graph construction ('mnn', 'scanorama', 'combat', or 'None'). 'None' means no batch removal."
    )
    ifa_group.add_argument(
        '--adj_dimensionality_reduction_method', type=str, default='PCA', choices=['PCA', 'autoencoder', 'nmf', 'None'],
        help="Dimensionality reduction method when 'adj_batch_removal_method' is not 'scanorama'. 'None' means no DR.")
    ifa_group.add_argument(
        '--adj_dim', type=int, default=30,
        help="Dimension for 'scanorama' or for dimensionality reduction if not 'scanorama' and DR method is not None.")
    ifa_group.add_argument(
        '--adj_scale', dest='adj_scale', action='store_true', default=True,
        help="Scale data before DR if 'adj_batch_removal_method' is not 'scanorama'. Use --no-adj-scale to disable.")
    ifa_group.add_argument('--no-adj-scale', dest='adj_scale', action='store_false',
                           help="Disable scaling for adjacency graph integration.")

    # --- inter_exp_adj_paras ---
    inter_ea_doc = """The module is used to construct the adjacency matrix of the expression graph, which
contains three subgraphs: a real-to-pseudo-spot graph, a pseudo-spots internal graph,
and a real-spots internal graph. Parameters for REAL-TO-PSEUDO graph."""
    inter_ea_group = parser.add_argument_group('Inter-Expression Adjacency (Real-to-Pseudo)', inter_ea_doc)
    inter_ea_group.add_argument('--inter_find_neighbor_method', type=str, default='MNN', choices=['MNN', 'KNN'],
                                help="Method for real-to-pseudo link graph construction ('MNN' or 'KNN').")
    inter_ea_group.add_argument('--inter_dist_method', type=str, default='cosine', choices=['euclidean', 'cosine'],
                                help="Distance metric for real-to-pseudo links ('euclidean' or 'cosine').")
    inter_ea_group.add_argument('--inter_corr_dist_neighbors', type=int, default=20,
                                help="Number of nearest neighbors for real-to-pseudo links.")

    # --- real_intra_exp_adj_paras ---
    real_intra_ea_group = parser.add_argument_group('Real Spot Intra-Expression Adjacency')
    real_intra_ea_group.add_argument('--real_intra_find_neighbor_method', type=str, default='MNN',
                                     choices=['MNN', 'KNN'],
                                     help="Method for real-spots internal graph construction ('MNN' or 'KNN').")
    real_intra_ea_group.add_argument('--real_intra_dist_method', type=str, default='cosine',
                                     choices=['euclidean', 'cosine'],
                                     help="Distance metric for real-spots internal graph ('euclidean' or 'cosine').")
    real_intra_ea_group.add_argument('--real_intra_corr_dist_neighbors', type=int, default=10,
                                     help="Number of nearest neighbors for real-spots internal graph.")
    real_intra_ea_group.add_argument(
        '--real_intra_pca_dimensionality_reduction', dest='real_intra_pca_dimensionality_reduction',
        action='store_true', default=False,
        help="Use PCA dimensionality reduction for real-spots internal graph. Disabled by default.")
    real_intra_ea_group.add_argument('--real_intra_dim', type=int, default=50,
                                     help="PCA dimension if 'real_intra_pca_dimensionality_reduction' is True.")

    # --- pseudo_intra_exp_adj_paras ---
    pseudo_intra_ea_group = parser.add_argument_group('Pseudo Spot Intra-Expression Adjacency')
    pseudo_intra_ea_group.add_argument('--pseudo_intra_find_neighbor_method', type=str, default='MNN',
                                       choices=['MNN', 'KNN'],
                                       help="Method for pseudo-spots internal graph construction ('MNN' or 'KNN').")
    pseudo_intra_ea_group.add_argument(
        '--pseudo_intra_dist_method', type=str, default='cosine', choices=['euclidean', 'cosine'],
        help="Distance metric for pseudo-spots internal graph ('euclidean' or 'cosine').")
    pseudo_intra_ea_group.add_argument('--pseudo_intra_corr_dist_neighbors', type=int, default=20,
                                       help="Number of nearest neighbors for pseudo-spots internal graph.")
    pseudo_intra_ea_group.add_argument(
        '--pseudo_intra_pca_dimensionality_reduction', dest='pseudo_intra_pca_dimensionality_reduction',
        action='store_true', default=False,
        help="Use PCA dimensionality reduction for pseudo-spots internal graph. Disabled by default.")
    pseudo_intra_ea_group.add_argument('--pseudo_intra_dim', type=int, default=50,
                                       help="PCA dimension if 'pseudo_intra_pca_dimensionality_reduction' is True.")

    # --- spatial_adj_paras ---
    sa_doc = """The module is used to construct the adjacency matrix of the spatial graph."""
    sa_group = parser.add_argument_group('Spatial Adjacency Parameters', sa_doc)
    sa_group.add_argument('--spatial_link_method', type=str, default='soft', choices=['soft', 'hard'],
                          help="Spatial graph link method ('soft' or 'hard').")
    sa_group.add_argument(
        '--space_dist_threshold',
        type=float,
        default=2.0,  # Can be None
        help="Maximum distance for spatial links. Set to 'None' for no threshold.")

    # --- integration_for_feature_paras ---
    iff_doc = """This module is used to integrate the normalized real- and pseudo- spots as the input
feature for STdGCN."""
    iff_group = parser.add_argument_group('Integration for Feature Parameters', iff_doc)
    iff_group.add_argument('--feat_batch_removal_method', type=str, default=None,
                           choices=['mnn', 'scanorama', 'combat', 'None'],
                           help="Batch removal method for STdGCN features ('mnn', 'scanorama', 'combat', or 'None').")
    iff_group.add_argument(
        '--feat_dimensionality_reduction_method', type=str, default=None, choices=['PCA', 'autoencoder', 'nmf', 'None'],
        help="Dimensionality reduction method for features if 'feat_batch_removal_method' is not 'scanorama'.")
    iff_group.add_argument(
        '--feat_dim', type=int, default=80,
        help="Dimension for 'scanorama' or for DR if not 'scanorama' and DR method is not None (for features).")
    iff_group.add_argument(
        '--feat_scale', dest='feat_scale', action='store_true', default=True, help=
        "Scale data for feature integration if 'feat_batch_removal_method' is not 'scanorama'. Use --no-feat-scale to disable."
    )
    iff_group.add_argument('--no-feat-scale', dest='feat_scale', action='store_false',
                           help="Disable scaling for feature integration.")

    # --- GCN_paras ---
    gcn_doc = """This module is used for setting the deep learning parameters for STdGCN."""
    gcn_group = parser.add_argument_group('GCN Parameters', gcn_doc)
    gcn_group.add_argument('--epoch_n', type=int, default=3000, help="Maximum number of GCN training epochs.")
    gcn_group.add_argument('--gcn_dim', type=int, default=80, help="Dimension of the GCN hidden layers.")
    gcn_group.add_argument('--common_hid_layers_num', type=int, default=1,
                           help="Number of common GCN layers = 'common_hid_layers_num'+1.")
    gcn_group.add_argument('--fcnn_hid_layers_num', type=int, default=1,
                           help="Number of fully connected neural network layers = 'fcnn_hid_layers_num'+2.")
    gcn_group.add_argument('--dropout', type=float, default=0.0, help="Dropout probability.")
    gcn_group.add_argument('--learning_rate_sgd', type=float, default=2e-1, help="Initial learning rate for SGD.")
    gcn_group.add_argument('--weight_decay_sgd', type=float, default=3e-4, help="L2 penalty for SGD.")
    gcn_group.add_argument('--momentum', type=float, default=0.9, help="Momentum factor for SGD.")
    gcn_group.add_argument('--dampening', type=float, default=0.0, help="Dampening for momentum in SGD.")
    gcn_group.add_argument('--nesterov', dest='nesterov', action='store_true', default=True,
                           help="Enables Nesterov momentum for SGD. Use --no-nesterov to disable.")
    gcn_group.add_argument('--no-nesterov', dest='nesterov', action='store_false',
                           help="Disable Nesterov momentum for SGD.")
    gcn_group.add_argument('--early_stopping_patience', type=int, default=20,
                           help="Number of epochs for early stopping patience.")
    gcn_group.add_argument('--clip_grad_max_norm', type=float, default=1.0,
                           help="Clips gradient norm of an iterable of parameters.")
    gcn_group.add_argument('--print_loss_epoch_step', type=int, default=20,
                           help="Print loss value at every 'print_loss_epoch_step' epoch.")

    # --- Main STdGCN run parameters ---
    run_doc = """General parameters for running STdGCN."""
    run_group = parser.add_argument_group('STdGCN Run Parameters', run_doc)
    run_group.add_argument('--load_test_groundtruth', dest='load_test_groundtruth', action='store_true', default=False,
                           help="Upload ground truth (ST_ground_truth.tsv) to track performance. Disabled by default.")
    run_group.add_argument(
        '--use_marker_genes', dest='use_marker_genes', action='store_true', default=True,
        help="Use gene selection process before running STdGCN. Use --no-use-marker-genes to disable (use common genes)."
    )
    run_group.add_argument('--no-use-marker-genes', dest='use_marker_genes', action='store_false',
                           help="Do not use marker genes; use common genes from SC and ST data.")
    run_group.add_argument(
        '--external_genes', dest='external_genes', action='store_true', default=False,
        help="When 'use_marker_genes'=True, upload a specified gene list (marker_genes.tsv). Disabled by default.")
    run_group.add_argument(
        '--generate_new_pseudo_spots', dest='generate_new_pseudo_spots', action='store_true', default=True, help=
        "Generate new pseudo-spots. If disabled, loads from 'pseudo_ST.pkl'. Use --no-generate-new-pseudo-spots to disable."
    )
    run_group.add_argument('--no-generate-new-pseudo-spots', dest='generate_new_pseudo_spots', action='store_false',
                           help="Do not generate new pseudo-spots; load from 'pseudo_ST.pkl' in output_path.")
    run_group.add_argument(
        '--fraction_pie_plot', dest='fraction_pie_plot', action='store_true', default=False,
        help="Draw pie plot of predicted results. Not recommended for large spot numbers. Disabled by default.")
    run_group.add_argument(
        '--cell_type_distribution_plot', dest='cell_type_distribution_plot', action='store_true', default=True, help=
        "Draw scatter plot of predicted results for each cell type. Use --no-cell-type-distribution-plot to disable.")
    run_group.add_argument('--no-cell-type-distribution-plot', dest='cell_type_distribution_plot', action='store_false',
                           help="Disable drawing scatter plot of predicted results for each cell type.")
    run_group.add_argument('--n_jobs', type=int, default=-1,
                           help="Number of threads for intraop parallelism on CPU (-1 for all CPUs).")
    run_group.add_argument('--gcn_device', type=str, default='GPU', choices=['GPU', 'CPU'],
                           help="Device to run GCN networks ('GPU' or 'CPU').")

    # Note on space_dist_threshold being None:
    # argparse by default will pass the default value if not specified.
    # If you want to allow 'None' as a string to signify Python None, you'd need custom type.
    # Here, if not provided, it's 2.0. To make it None, one might need to omit the default
    # or use a specific string like "disable" and handle it post-parsing.
    # For simplicity, I've kept it as float default 2.0. The user can set a very large number
    # if they want to effectively disable it, or the code using it can check for a specific value like 0 or -1.
    # The original doc says "[float or None]", which `type=float, default=None` handles well if default is None.
    # Here default is 2.0, so it's always a float.
    # If you need to explicitly pass None:
    # A common way is to have `default="2.0"` and then a post-processing step:
    # `if args.space_dist_threshold == "None": args.space_dist_threshold = None else: args.space_dist_threshold = float(args.space_dist_threshold)`
    # Or, don't provide a default in argparse, and set it in your code if `args.space_dist_threshold is None`.
    # Given the original `spatial_adj_paras` has `2`, I'll stick to `default=2.0`.

    return parser


if __name__ == "__main__":
    arg_parser = parse_arguments()
    args = arg_parser.parse_args()
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    file_root_path = Path(args.root_path, args.dataset).resolve()
    pipeline_planer = PipelinePlaner.from_config_file(f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))
        set_seed(args.seed)
        stdgcnwrapper = stdGCNWrapper(cli_args=args)
        dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
        # preprocessing_pipeline=stdGCNWrapper.preprocessing_pipeline(args)
        data = dataset.load_data()
        kwargs = {tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)
        ST_adata_filter_norm = data.get_split_data(split_name="test")
        pseudo_adata_norm = data.get_split_data(split_name="pseudo")
        ST_adata_filter = data.data.raw.to_adata()[data.get_split_idx("test")]
        pseudo_adata_filter = data.data.raw.to_adata()[data.get_split_idx("pseudo")]
        ST_integration_batch_removed = data.data.obsm["DataInteragraionTransform"]
        adj_exp = data.data.uns['adj_exp']
        adj_sp = data.data.uns['adj_sp']
        word_to_idx_celltype = data.data.uns['word_to_idx_celltype']
        train_valid_len = pseudo_adata_filter.shape[0]
        test_len = ST_adata_filter.shape[0]
        train_idx, valid_idx = train_test_split(np.arange(train_valid_len) + test_len, test_size=0.3)
        test_idx = np.arange(test_len)
        table1 = ST_adata_filter_norm.obsm['cell_type_portion'].copy()
        label1 = pd.concat([table1, pseudo_adata_filter.obs.iloc[:, :-1]])[pseudo_adata_filter.uns['cell_types_list']]
        # label1 = table1[pseudo_adata_filter.obs.iloc[:, :-1].columns].append(pseudo_adata_filter.obs.iloc[:, :-1])
        label1 = torch.tensor(label1.values.astype(float))
        inputs = (ST_adata_filter_norm, ST_integration_batch_removed, adj_exp, adj_sp, word_to_idx_celltype, train_idx,
                  valid_idx, test_idx)
        # ref_count = data.get_feature(split_name="ref", return_type="numpy")
        # ref_annot = data.get_feature(split_name="ref", return_type="numpy", channel="cellType", channel_type="obs")

        valid_score, test_score = stdgcnwrapper.fit_score(x=inputs, y=label1, valid_idx=valid_idx, test_idx=test_idx)
        wandb.log({"MSE": valid_score, "test_MSE": test_score})
        # stdgcnwrapper.fit()
        # results = stdgcnwrapper.predict()
        # results.write_h5ad(paths['output_path']+'/results.h5ad')

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  #Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(
            result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
            conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
            root_path=file_root_path, ascending=True, required_funs=[
                "CelltypeTransform", "pseudoSpotGen", "RemoveSplit", "FilterGenesCommon", "SaveRaw",
                "updateAnndataObsTransform", "CellTypeNum", "DataInteragraionTransform", "stdgcnGraph",
                "DataInteragraionTransform", "SetConfig"
            ], required_indexes=[0, 1, 2, 3, 5, 8, 9, 11, 12, 13, 14], metric="MSE")
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
    """
    python stdgcn.py  --feat_dimensionality_reduction_method None  --no-feat-scale --adj_dimensionality_reduction_method None --no-adj-scale
    """
