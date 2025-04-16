#!/usr/bin/env python

import argparse  # Import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from dance.modules.spatial.cell_type_deconvo.stdgcn import (
    combine_and_normalize_adj,
    conGCN_train,
    find_or_load_marker_genes,
    generate_or_load_pseudo_spots,
    load_data,
    prepare_adjacency_matrices,
    prepare_feature_matrix,
    preprocess_data,
    save_and_plot_results,
    setup_gcn_components,
)


def train_gcn_model(model, feature, adjs, labels, train_valid_len, test_len, GCN_paras, optimizer, scheduler, loss_fn,
                    load_test_groundtruth, GCN_device, n_jobs):
    """Trains the GCN model."""
    print("Starting GCN training...")
    output, loss_history, trained_model = conGCN_train(
        model=model,
        train_valid_len=train_valid_len,
        train_valid_ratio=0.9,  # Hardcoded, consider making a parameter
        test_len=test_len,
        feature=feature,
        adjs=adjs,
        label=labels,
        epoch_n=GCN_paras['epoch_n'],
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=GCN_paras['early_stopping_patience'],
        clip_grad_max_norm=GCN_paras['clip_grad_max_norm'],
        load_test_groundtruth=load_test_groundtruth,
        print_epoch_step=GCN_paras['print_loss_epoch_step'],
        cpu_num=n_jobs,
        GCN_device=GCN_device)
    print("GCN training finished.")
    return output, loss_history, trained_model


def run_STdGCN(paths, find_marker_genes_paras, pseudo_spot_simulation_paras, data_normalization_paras,
               integration_for_adj_paras, inter_exp_adj_paras, spatial_adj_paras, real_intra_exp_adj_paras,
               pseudo_intra_exp_adj_paras, integration_for_feature_paras, GCN_paras, load_test_groundtruth=False,
               use_marker_genes=True, external_genes=False, generate_new_pseudo_spots=True, fraction_pie_plot=False,
               cell_type_distribution_plot=True, n_jobs=-1, GCN_device='CPU'):
    """Runs the STdGCN pipeline by calling modularized functions.

    Parameters mirror the original function and argparse setup.

    """
    sc_path = paths['sc_path']
    st_path = paths['ST_path']
    output_path = paths['output_path']
    os.makedirs(output_path, exist_ok=True)  # Ensure output dir exists

    # 1. Load Data
    sc_adata, st_adata, cell_types, word_to_idx_celltype, idx_to_word_celltype = load_data(
        sc_path, st_path, load_test_groundtruth)

    # 2. Find/Load Marker Genes
    selected_genes = find_or_load_marker_genes(sc_adata, sc_path, output_path, use_marker_genes, external_genes,
                                               find_marker_genes_paras)

    # 3. Generate/Load Pseudo-spots
    pseudo_adata = generate_or_load_pseudo_spots(sc_adata, output_path, generate_new_pseudo_spots,
                                                 pseudo_spot_simulation_paras, idx_to_word_celltype, n_jobs)

    # 4. Preprocess Data (Filter genes, Normalize, Scale)
    st_adata_norm, pseudo_adata_norm = preprocess_data(st_adata, pseudo_adata, selected_genes, data_normalization_paras,
                                                       cell_types)

    # 5. Prepare Adjacency Matrices (Raw)
    A_inter_exp, A_intra_space, A_real_intra_exp, A_pseudo_intra_exp = prepare_adjacency_matrices(
        st_adata_norm, pseudo_adata_norm, integration_for_adj_paras, inter_exp_adj_paras, real_intra_exp_adj_paras,
        pseudo_intra_exp_adj_paras, spatial_adj_paras, n_jobs, GCN_device)

    # 6. Combine and Normalize Adjacency Matrices
    real_num = st_adata_norm.shape[0]
    pseudo_num = pseudo_adata_norm.shape[0]
    adj_exp_final, adj_sp_final = combine_and_normalize_adj(
        A_inter_exp,
        A_real_intra_exp,
        A_pseudo_intra_exp,
        A_intra_space,
        real_num,
        pseudo_num,
        adj_alpha=1,
        adj_beta=1,
        diag_power=20,
        normalize=True  # Keep params or make configurable
    )
    adjs = [adj_exp_final, adj_sp_final]  # List of final adjacency tensors

    # 7. Prepare Feature Matrix
    feature = prepare_feature_matrix(st_adata_norm, pseudo_adata_norm, integration_for_feature_paras, n_jobs,
                                     GCN_device)

    # 8. Setup GCN Model, Optimizer, Scheduler, Loss
    input_layer_dim = feature.shape[1]
    hidden_layer_dim = min(int(st_adata_norm.shape[1] * 1 / 2), GCN_paras['dim'])  # Logic from original
    output_layer_dim = len(cell_types)

    model, optimizer, scheduler, loss_fn = setup_gcn_components(input_layer_dim, hidden_layer_dim, output_layer_dim,
                                                                GCN_paras)

    # 9. Prepare Labels for Training
    # Use cell type fractions from pseudo-spots as labels
    # Need to handle ST spot labels (if groundtruth exists or use pseudo-labels?)
    # Original code appends ST obs (potentially GT or zeros) to pseudo obs
    st_labels_df = st_adata_norm.obs[list(cell_types)]  # Get GT or zeros added in preprocess_data
    pseudo_labels_df = pseudo_adata_norm.obs[list(cell_types)]  # Get fractions from pseudo generation
    combined_labels_df = pd.concat([st_labels_df, pseudo_labels_df], axis=0)
    # Ensure order matches the feature matrix order (assuming concat(ST, Pseudo))
    labels_tensor = torch.tensor(combined_labels_df.values).float()  # Ensure float

    # 10. Train GCN Model
    train_valid_len = pseudo_num  # Train/Validation split happens on pseudo-spots
    test_len = real_num  # Test predictions are for real spots
    predictions_raw, loss_history, trained_model = train_gcn_model(model, feature, adjs, labels_tensor, train_valid_len,
                                                                   test_len, GCN_paras, optimizer, scheduler, loss_fn,
                                                                   load_test_groundtruth, GCN_device, n_jobs)

    # 11. Save Results and Plot
    # Extract the predictions for the real ST spots only
    st_predictions_raw = predictions_raw[:test_len]

    # Get column names used for pseudo labels (should match cell_types)
    pseudo_adata_obs_cols = list(pseudo_adata_norm.obs[list(cell_types)].columns)

    save_and_plot_results(output_path, st_adata_norm, trained_model, loss_history, st_predictions_raw, cell_types,
                          pseudo_adata_obs_cols, load_test_groundtruth, fraction_pie_plot, cell_type_distribution_plot)

    # 12. Add predictions to the final AnnData object and return
    st_adata_norm.obsm['predict_result'] = np.exp(st_predictions_raw.detach().cpu().numpy())

    # Optional: Clean up GPU memory
    if GCN_device == 'GPU' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("STdGCN run finished successfully.")
    return st_adata_norm


def parse_args():
    parser = argparse.ArgumentParser(description="Run STdGCN for spatial transcriptomics deconvolution.")

    # --- Path Parameters ---
    path_group = parser.add_argument_group('Input/Output Paths')
    path_group.add_argument('--sc_path', type=str, default='./data/sc_data',
                            help='Path for loading single cell reference data. Expects sc_data.tsv and sc_label.tsv.')
    path_group.add_argument(
        '--ST_path', type=str, default='./data/ST_data',
        help='Path for loading spatial transcriptomics data. Expects ST_data.tsv and coordinates.csv.')
    path_group.add_argument('--output_path', type=str, default='./output', help='Path for saving output files.')
    path_group.add_argument(
        '--marker_genes_file', type=str, default='marker_genes.tsv',
        help='[Optional] Name of the marker gene list file within sc_path. Default: marker_genes.tsv')
    path_group.add_argument(
        '--ground_truth_file', type=str, default='ST_ground_truth.tsv',
        help='[Optional] Name of the ST ground truth file within ST_path. Default: ST_ground_truth.tsv')

    # --- Marker Gene Finding Parameters ---
    marker_group = parser.add_argument_group('Marker Gene Finding Parameters')
    # Use --no-preprocess to disable preprocessing
    marker_group.add_argument(
        '--preprocess-markers', dest='preprocess_markers', action='store_true', default=True, help=
        'Preprocess input expression data for marker finding (normalize, log, hvg, regress out, scale). Enabled by default.'
    )
    marker_group.add_argument('--no-preprocess-markers', dest='preprocess_markers', action='store_false',
                              help='Disable preprocessing for marker finding.')
    # Conditional defaults based on preprocess_markers being True
    marker_group.add_argument(
        '--marker-normalize', dest='marker_normalize', action='store_true', default=True,
        help='Normalize scRNA-seq data for marker finding (if --preprocess-markers). Enabled by default.')
    marker_group.add_argument('--no-marker-normalize', dest='marker_normalize', action='store_false',
                              help='Disable normalization for marker finding.')
    marker_group.add_argument(
        '--marker-log', dest='marker_log', action='store_true', default=True,
        help='Logarithmize scRNA-seq data for marker finding (if --preprocess-markers). Enabled by default.')
    marker_group.add_argument('--no-marker-log', dest='marker_log', action='store_false',
                              help='Disable logarithmization for marker finding.')
    marker_group.add_argument(
        '--highly-variable-genes', dest='highly_variable_genes', action='store_true', default=False,
        help='Filter highly variable genes for marker finding (if --preprocess-markers). Disabled by default.')
    marker_group.add_argument(
        '--highly-variable-gene-num', type=int, default=None,
        help='Number of highly-variable genes to keep (if --highly-variable-genes). Default: None (uses scanpy default).'
    )
    marker_group.add_argument(
        '--regress-out', dest='regress_out', action='store_true', default=False,
        help='Regress out mitochondrial genes for marker finding (if --preprocess-markers). Disabled by default.')
    marker_group.add_argument(
        '--marker-scale', dest='marker_scale', action='store_true', default=False,
        help='Scale scRNA-seq data for marker finding (if --preprocess-markers). Disabled by default.')
    marker_group.add_argument('--marker-pca-components', type=int, default=30,
                              help='Number of PCA components for marker finding (used by some methods/visualizations).')
    marker_group.add_argument('--marker-gene-method', type=str, default='logreg', choices=['logreg', 'wilcoxon'],
                              help='Method for identifying marker genes.')
    marker_group.add_argument('--top-gene-per-type', type=int, default=100,
                              help='Number of top marker genes per cell type to select.')
    # Wilcoxon specific filtering
    marker_group.add_argument('--filter-wilcoxon-marker-genes', dest='filter_wilcoxon_marker_genes',
                              action='store_true', default=True,
                              help='Apply additional filters if marker_gene_method is wilcoxon. Enabled by default.')
    marker_group.add_argument('--no-filter-wilcoxon-marker-genes', dest='filter_wilcoxon_marker_genes',
                              action='store_false', help='Disable additional wilcoxon marker gene filters.')
    marker_group.add_argument('--pvals-adj-threshold', type=float, default=0.10,
                              help='Corrected p-value threshold for Wilcoxon marker filtering.')
    marker_group.add_argument('--log-fold-change-threshold', type=float, default=1.0,
                              help='Log fold change threshold for Wilcoxon marker filtering.')
    marker_group.add_argument('--min-within-group-fraction-threshold', type=float, default=None,
                              help='Min fraction expression within cell type for Wilcoxon marker filtering.')
    marker_group.add_argument('--max-between-group-fraction-threshold', type=float, default=None,
                              help='Max fraction expression outside cell type for Wilcoxon marker filtering.')

    # --- Pseudo-Spot Simulation Parameters ---
    pseudo_group = parser.add_argument_group('Pseudo-Spot Simulation Parameters')
    pseudo_group.add_argument('--pseudo-spot-num', type=int, default=30000, help='Number of pseudo-spots to simulate.')
    pseudo_group.add_argument('--min-cell-num-in-spot', type=int, default=8,
                              help='Minimum number of cells in a pseudo-spot.')
    pseudo_group.add_argument('--max-cell-num-in-spot', type=int, default=12,
                              help='Maximum number of cells in a pseudo-spot.')
    pseudo_group.add_argument('--pseudo-generation-method', type=str, default='celltype', choices=['cell', 'celltype'],
                              help='Method for simulating pseudo-spots (\'cell\' or \'celltype\').')
    pseudo_group.add_argument('--max-cell-types-in-spot', type=int, default=4,
                              help='Maximum number of cell types in a pseudo-spot (if generation_method=\'celltype\').')

    # --- Data Normalization Parameters (Post-Simulation) ---
    norm_group = parser.add_argument_group('Data Normalization Parameters (Post-Simulation)')
    norm_group.add_argument('--data-normalize', dest='data_normalize', action='store_true', default=True,
                            help='Normalize real and pseudo spots (total counts = 10k). Enabled by default.')
    norm_group.add_argument('--no-data-normalize', dest='data_normalize', action='store_false',
                            help='Disable normalization of real/pseudo spots.')
    norm_group.add_argument('--data-log', dest='data_log', action='store_true', default=True,
                            help='Logarithmize real and pseudo spots (log(X+1)). Enabled by default.')
    norm_group.add_argument('--no-data-log', dest='data_log', action='store_false',
                            help='Disable logarithmization of real/pseudo spots.')
    norm_group.add_argument('--data-scale', dest='data_scale', action='store_true', default=False,
                            help='Scale real and pseudo spots (unit variance, zero mean). Disabled by default.')

    # --- Integration for Adjacency Matrix Parameters ---
    adj_int_group = parser.add_argument_group('Integration for Adjacency Matrix Parameters')
    adj_int_group.add_argument('--adj-batch-removal-method', type=str, default='none',
                               choices=['mnn', 'scanorama', 'combat', 'none'],
                               help='Batch removal method for adjacency matrix construction. Default: none.')
    adj_int_group.add_argument(
        '--adj-dim', type=int, default=30,
        help='Dimension for batch removal or dimensionality reduction for adjacency construction.')
    adj_int_group.add_argument(
        '--adj-dim-reduction-method', type=str, default='PCA', choices=['PCA', 'autoencoder', 'nmf', 'none'],
        help='Dimensionality reduction method for adjacency construction (if not scanorama). Default: PCA.')
    adj_int_group.add_argument(
        '--adj-scale', dest='adj_scale', action='store_true', default=True, help=
        'Scale data before dimensionality reduction for adjacency construction (if not scanorama). Enabled by default.')
    adj_int_group.add_argument('--no-adj-scale', dest='adj_scale', action='store_false',
                               help='Disable scaling for adjacency construction.')

    # --- Inter-Spot Expression Graph Parameters (Real-to-Pseudo) ---
    inter_exp_group = parser.add_argument_group('Inter-Spot Expression Graph Parameters (Real-to-Pseudo)')
    inter_exp_group.add_argument('--inter-exp-neighbor-method', type=str, default='MNN', choices=['MNN', 'KNN'],
                                 help='Neighbor finding method for real-to-pseudo graph.')
    inter_exp_group.add_argument('--inter-exp-dist-method', type=str, default='cosine', choices=['euclidean', 'cosine'],
                                 help='Distance metric for real-to-pseudo graph.')
    inter_exp_group.add_argument('--inter-exp-neighbors', type=int, default=20,
                                 help='Number of neighbors for real-to-pseudo graph.')

    # --- Real Intra-Spot Expression Graph Parameters ---
    real_intra_exp_group = parser.add_argument_group('Real Intra-Spot Expression Graph Parameters')
    real_intra_exp_group.add_argument('--real-intra-exp-neighbor-method', type=str, default='MNN',
                                      choices=['MNN',
                                               'KNN'], help='Neighbor finding method for real-spot internal graph.')
    real_intra_exp_group.add_argument('--real-intra-exp-dist-method', type=str, default='cosine',
                                      choices=['euclidean',
                                               'cosine'], help='Distance metric for real-spot internal graph.')
    real_intra_exp_group.add_argument('--real-intra-exp-neighbors', type=int, default=10,
                                      help='Number of neighbors for real-spot internal graph.')
    real_intra_exp_group.add_argument(
        '--real-intra-exp-pca', dest='real_intra_exp_pca', action='store_true', default=False, help=
        'Use PCA dimensionality reduction before computing distances for real-spot internal graph. Disabled by default.'
    )
    real_intra_exp_group.add_argument('--real-intra-exp-pca-dim', type=int, default=50,
                                      help='PCA dimension for real-spot internal graph (if --real-intra-exp-pca).')

    # --- Pseudo Intra-Spot Expression Graph Parameters ---
    pseudo_intra_exp_group = parser.add_argument_group('Pseudo Intra-Spot Expression Graph Parameters')
    pseudo_intra_exp_group.add_argument('--pseudo-intra-exp-neighbor-method', type=str, default='MNN',
                                        choices=['MNN',
                                                 'KNN'], help='Neighbor finding method for pseudo-spot internal graph.')
    pseudo_intra_exp_group.add_argument('--pseudo-intra-exp-dist-method', type=str, default='cosine',
                                        choices=['euclidean',
                                                 'cosine'], help='Distance metric for pseudo-spot internal graph.')
    pseudo_intra_exp_group.add_argument('--pseudo-intra-exp-neighbors', type=int, default=20,
                                        help='Number of neighbors for pseudo-spot internal graph.')
    pseudo_intra_exp_group.add_argument(
        '--pseudo-intra-exp-pca', dest='pseudo_intra_exp_pca', action='store_true', default=False, help=
        'Use PCA dimensionality reduction before computing distances for pseudo-spot internal graph. Disabled by default.'
    )
    pseudo_intra_exp_group.add_argument(
        '--pseudo-intra-exp-pca-dim', type=int, default=50,
        help='PCA dimension for pseudo-spot internal graph (if --pseudo-intra-exp-pca).')

    # --- Spatial Graph Parameters ---
    spatial_group = parser.add_argument_group('Spatial Graph Parameters')
    spatial_group.add_argument('--spatial-link-method', type=str, default='soft', choices=['soft', 'hard'],
                               help='Method for linking spots in the spatial graph (\'soft\'=1/dist, \'hard\'=1).')
    spatial_group.add_argument(
        '--spatial-dist-threshold', type=float, default=2.0,
        help='Maximum distance for linking spots in the spatial graph. Use negative value or inf for no threshold.')

    # --- Integration for Feature Matrix Parameters ---
    feat_int_group = parser.add_argument_group('Integration for Feature Matrix Parameters')
    feat_int_group.add_argument('--feature-batch-removal-method', type=str, default='none',
                                choices=['mnn', 'scanorama', 'combat',
                                         'none'], help='Batch removal method for final feature matrix. Default: none.')
    feat_int_group.add_argument(
        '--feature-dim-reduction-method', type=str, default='none', choices=['PCA', 'autoencoder', 'nmf', 'none'],
        help='Dimensionality reduction method for final feature matrix (if not scanorama). Default: none.')
    feat_int_group.add_argument('--feature-dim', type=int, default=80,
                                help='Dimension for batch removal or dimensionality reduction for feature matrix.')
    feat_int_group.add_argument(
        '--feature-scale', dest='feature_scale', action='store_true', default=True,
        help='Scale data before dimensionality reduction for feature matrix (if not scanorama). Enabled by default.')
    feat_int_group.add_argument('--no-feature-scale', dest='feature_scale', action='store_false',
                                help='Disable scaling for feature matrix construction.')

    # --- GCN Training Parameters ---
    gcn_group = parser.add_argument_group('GCN Training Parameters')
    gcn_group.add_argument('--epochs', type=int, default=3000, help='Maximum number of training epochs.')
    gcn_group.add_argument('--gcn-hidden-dim', type=int, default=80, help='Dimension of GCN hidden layers.')
    gcn_group.add_argument('--gcn-common-layers', type=int, default=1,
                           help='Number of common GCN hidden layers (total GCN layers = this + 1).')
    gcn_group.add_argument('--fcnn-layers', type=int, default=1,
                           help='Number of FCNN hidden layers (total FCNN layers = this + 2).')
    gcn_group.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    gcn_group.add_argument('--lr', type=float, default=2e-1, help='Initial learning rate for SGD.')
    gcn_group.add_argument('--weight-decay', type=float, default=3e-4, help='L2 penalty (weight decay) for SGD.')
    gcn_group.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD.')
    gcn_group.add_argument('--dampening', type=float, default=0.0, help='Dampening for momentum for SGD.')
    gcn_group.add_argument('--nesterov', dest='nesterov', action='store_true', default=True,
                           help='Enables Nesterov momentum for SGD. Enabled by default.')
    gcn_group.add_argument('--no-nesterov', dest='nesterov', action='store_false', help='Disable Nesterov momentum.')
    gcn_group.add_argument('--early-stopping-patience', type=int, default=20,
                           help='Number of epochs with no improvement to wait before early stopping.')
    gcn_group.add_argument('--clip-grad-max-norm', type=float, default=1.0, help='Max norm for gradient clipping.')
    gcn_group.add_argument('--print-loss-step', type=int, default=20, help='Print loss every N epochs.')

    # --- Run Control Parameters ---
    run_group = parser.add_argument_group('Run Control Parameters')
    run_group.add_argument(
        '--load-test-groundtruth', action='store_true', default=False,
        help='Load ground truth file (specified by --ground_truth_file in --ST_path) to track performance.')
    # Default is True for using marker genes
    run_group.add_argument('--use-marker-genes', dest='use_marker_genes', action='store_true', default=True,
                           help='Use marker gene selection process. Enabled by default.')
    run_group.add_argument('--no-marker-genes', dest='use_marker_genes', action='store_false',
                           help='Use common genes instead of marker gene selection.')
    run_group.add_argument(
        '--external-genes', action='store_true', default=False, help=
        'Use an external marker gene list (specified by --marker_genes_file in --sc_path) instead of calculating them.')
    # Default is True for generating new spots
    run_group.add_argument('--generate-new-pseudo-spots', dest='generate_new_pseudo_spots', action='store_true',
                           default=True, help='Generate new pseudo-spots. Enabled by default.')
    run_group.add_argument(
        '--no-generate-new-pseudo-spots', dest='generate_new_pseudo_spots', action='store_false',
        help='Do not generate new pseudo-spots (attempts to load from pseudo_ST.pkl in output_path).')
    # Default is True for pie plot
    run_group.add_argument(
        '--fraction-pie-plot', dest='fraction_pie_plot', action='store_true', default=True,
        help='Generate pie plots of predicted fractions (can be slow for many spots). Enabled by default.')
    run_group.add_argument('--no-fraction-pie-plot', dest='fraction_pie_plot', action='store_false',
                           help='Disable pie plot generation.')
    # Default is True for scatter plot
    run_group.add_argument('--cell-type-distribution-plot', dest='cell_type_distribution_plot', action='store_true',
                           default=True,
                           help='Generate scatter plots of predicted cell type distributions. Enabled by default.')
    run_group.add_argument('--no-cell-type-distribution-plot', dest='cell_type_distribution_plot', action='store_false',
                           help='Disable cell type distribution plot generation.')
    run_group.add_argument('--n-jobs', type=int, default=-1,
                           help='Number of CPU threads for parallel tasks (-1 uses all available).')
    run_group.add_argument('--gcn-device', type=str, default='GPU', choices=['GPU', 'CPU'],
                           help='Device to run GCN training on.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Reconstruct the dictionaries from args
    # Handle 'none' string conversion back to None for relevant parameters
    def none_or_str(value):
        return None if value.lower() == 'none' else value

    def none_or_float(value):
        return None if value is None else float(value)

    def none_or_int(value):
        return None if value is None else int(value)

    paths = {
        'sc_path': args.sc_path,
        'ST_path': args.ST_path,
        'output_path': args.output_path,
        # Optional files - might need adjustment in run_STdGCN to use these names directly
        # 'marker_genes_file': args.marker_genes_file,
        # 'ground_truth_file': args.ground_truth_file,
    }
    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)

    find_marker_genes_paras = {
        'preprocess': args.preprocess_markers,
        'normalize': args.marker_normalize,
        'log': args.marker_log,
        'highly_variable_genes': args.highly_variable_genes,
        'highly_variable_gene_num': none_or_int(args.highly_variable_gene_num),
        'regress_out': args.regress_out,
        'scale': args.marker_scale,
        'PCA_components': args.marker_pca_components,
        'marker_gene_method': args.marker_gene_method,
        'top_gene_per_type': args.top_gene_per_type,
        'filter_wilcoxon_marker_genes': args.filter_wilcoxon_marker_genes,
        'pvals_adj_threshold': none_or_float(args.pvals_adj_threshold),
        'log_fold_change_threshold': none_or_float(args.log_fold_change_threshold),
        'min_within_group_fraction_threshold': none_or_float(args.min_within_group_fraction_threshold),
        'max_between_group_fraction_threshold': none_or_float(args.max_between_group_fraction_threshold),
    }

    pseudo_spot_simulation_paras = {
        'spot_num': args.pseudo_spot_num,
        'min_cell_num_in_spot': args.min_cell_num_in_spot,
        'max_cell_num_in_spot': args.max_cell_num_in_spot,
        'generation_method': args.pseudo_generation_method,
        'max_cell_types_in_spot': args.max_cell_types_in_spot,
    }

    data_normalization_paras = {
        'normalize': args.data_normalize,
        'log': args.data_log,
        'scale': args.data_scale,
    }

    integration_for_adj_paras = {
        'batch_removal_method': none_or_str(args.adj_batch_removal_method),
        'dim': args.adj_dim,
        'dimensionality_reduction_method': none_or_str(args.adj_dim_reduction_method),
        'scale': args.adj_scale,
    }

    inter_exp_adj_paras = {
        'find_neighbor_method': args.inter_exp_neighbor_method,
        'dist_method': args.inter_exp_dist_method,
        'corr_dist_neighbors': args.inter_exp_neighbors,
    }

    real_intra_exp_adj_paras = {
        'find_neighbor_method': args.real_intra_exp_neighbor_method,
        'dist_method': args.real_intra_exp_dist_method,
        'corr_dist_neighbors': args.real_intra_exp_neighbors,
        'PCA_dimensionality_reduction': args.real_intra_exp_pca,
        'dim': args.real_intra_exp_pca_dim,
    }

    pseudo_intra_exp_adj_paras = {
        'find_neighbor_method': args.pseudo_intra_exp_neighbor_method,
        'dist_method': args.pseudo_intra_exp_dist_method,
        'corr_dist_neighbors': args.pseudo_intra_exp_neighbors,
        'PCA_dimensionality_reduction': args.pseudo_intra_exp_pca,
        'dim': args.pseudo_intra_exp_pca_dim,
    }

    spatial_adj_paras = {
        'link_method': args.spatial_link_method,
        'space_dist_threshold': none_or_float(args.spatial_dist_threshold),
    }

    integration_for_feature_paras = {
        'batch_removal_method': none_or_str(args.feature_batch_removal_method),
        'dimensionality_reduction_method': none_or_str(args.feature_dim_reduction_method),
        'dim': args.feature_dim,
        'scale': args.feature_scale,
    }

    GCN_paras = {
        'epoch_n': args.epochs,
        'dim': args.gcn_hidden_dim,
        'common_hid_layers_num': args.gcn_common_layers,
        'fcnn_hid_layers_num': args.fcnn_layers,
        'dropout': args.dropout,
        'learning_rate_SGD': args.lr,
        'weight_decay_SGD': args.weight_decay,
        'momentum': args.momentum,
        'dampening': args.dampening,
        'nesterov': args.nesterov,
        'early_stopping_patience': args.early_stopping_patience,
        'clip_grad_max_norm': args.clip_grad_max_norm,
        'print_loss_epoch_step': args.print_loss_step,
    }

    # Note: You might need to slightly adjust how run_STdGCN handles optional file names
    # if it doesn't already look for default names like 'marker_genes.tsv' within the sc_path.
    # The current setup assumes run_STdGCN uses fixed filenames within the provided paths.
    # If you need to pass the filenames themselves, uncomment them in the `paths` dict
    # and modify run_STdGCN accordingly.

    print("--- Running STdGCN with Parameters ---")
    print(f"Paths: {paths}")
    print(f"Marker Gene Params: {find_marker_genes_paras}")
    print(f"Pseudo Spot Params: {pseudo_spot_simulation_paras}")
    print(f"Data Norm Params: {data_normalization_paras}")
    print(f"Adj Integration Params: {integration_for_adj_paras}")
    print(f"Inter Exp Adj Params: {inter_exp_adj_paras}")
    print(f"Real Intra Exp Adj Params: {real_intra_exp_adj_paras}")
    print(f"Pseudo Intra Exp Adj Params: {pseudo_intra_exp_adj_paras}")
    print(f"Spatial Adj Params: {spatial_adj_paras}")
    print(f"Feature Integration Params: {integration_for_feature_paras}")
    print(f"GCN Params: {GCN_paras}")
    print(f"Load Ground Truth: {args.load_test_groundtruth}")
    print(f"Use Marker Genes: {args.use_marker_genes}")
    print(f"Use External Genes: {args.external_genes}")
    print(f"Generate New Pseudo Spots: {args.generate_new_pseudo_spots}")
    print(f"Fraction Pie Plot: {args.fraction_pie_plot}")
    print(f"Cell Type Dist Plot: {args.cell_type_distribution_plot}")
    print(f"N Jobs: {args.n_jobs}")
    print(f"GCN Device: {args.gcn_device}")
    print("------------------------------------")
    # ==================================================================
    # Main Refactored Function
    # ==================================================================

    results = run_STdGCN(
        paths=paths,  # Pass the reconstructed dict
        load_test_groundtruth=args.load_test_groundtruth,
        use_marker_genes=args.use_marker_genes,
        external_genes=args.external_genes,
        find_marker_genes_paras=find_marker_genes_paras,  # Pass the reconstructed dict
        generate_new_pseudo_spots=args.generate_new_pseudo_spots,
        pseudo_spot_simulation_paras=pseudo_spot_simulation_paras,  # Pass the reconstructed dict
        data_normalization_paras=data_normalization_paras,  # Pass the reconstructed dict
        integration_for_adj_paras=integration_for_adj_paras,  # Pass the reconstructed dict
        inter_exp_adj_paras=inter_exp_adj_paras,  # Pass the reconstructed dict
        spatial_adj_paras=spatial_adj_paras,  # Pass the reconstructed dict
        real_intra_exp_adj_paras=real_intra_exp_adj_paras,  # Pass the reconstructed dict
        pseudo_intra_exp_adj_paras=pseudo_intra_exp_adj_paras,  # Pass the reconstructed dict
        integration_for_feature_paras=integration_for_feature_paras,  # Pass the reconstructed dict
        GCN_paras=GCN_paras,  # Pass the reconstructed dict
        fraction_pie_plot=args.fraction_pie_plot,
        cell_type_distribution_plot=args.cell_type_distribution_plot,
        n_jobs=args.n_jobs,
        GCN_device=args.gcn_device)

    # Save results (assuming run_STdGCN returns an object with write_h5ad)
    if results:
        results_path = os.path.join(args.output_path, 'results.h5ad')
        print(f"Saving results to {results_path}")
        results.write_h5ad(results_path)
    else:
        print("run_STdGCN did not return results.")


if __name__ == '__main__':
    main()
