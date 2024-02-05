import argparse
from pprint import pformat

import numpy as np
import scanpy as sc

from dance import logger
from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.scgnn2 import ScGNN2
from dance.transforms import (
    AnnDataTransform,
    CellwiseMaskData,
    Compose,
    FilterCellsScanpy,
    FilterGenesScanpy,
    FilterGenesTopK,
)
from dance.utils import set_seed

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Main program for scGNN v2")

    # Program related
    parser.add_argument(
        "--use_bulk", action="store_true", default=False,
        help="(boolean, default False) If True, expect a bulk expression file and will run "
        "deconvolution and imputation")
    parser.add_argument(
        "--given_cell_type_labels", action="store_true", default=False,
        help="(boolean, default False) If True, expect a cell type label file and will compute ARI "
        "against those labels")
    parser.add_argument("--run_LTMG", action="store_true", default=False,
                        help="(boolean, default False) Not fully implemented")
    parser.add_argument("--use_CCC", action="store_true", default=False,
                        help="(boolean, default False) Not fully implemented")
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1,
        help="(float, default 0.1) Probability that a non-zero value in the sc expression matrix will "
        "be set to zero. If this is set to 0, will not perform dropout or compute imputation error ")
    parser.add_argument("--total_epoch", type=int, default=31, help="(int, default 10) Total EM epochs")
    parser.add_argument("--ari_threshold", type=float, default=0.95, help="(float, default 0.95) The threshold for ari")
    parser.add_argument("--graph_change_threshold", type=float, default=0.01,
                        help="(float, default 0.01) The threshold for graph change")
    parser.add_argument("--alpha", type=float, default=0.5, help="(float, default 0.5)")

    # Data loading related
    parser.add_argument(
        "--load_dataset_dir", type=str, default="/fs/ess/PCON0022/Edison/datasets",
        help="(str) Folder that stores all your datasets. For example, if your expression matrix is in "
        "/fs/ess/PCON1234/Brutus/datasets/12.Klein/T2000_expression.csv, this should be set to "
        "/fs/ess/PCON1234/Brutus/datasets")
    parser.add_argument(
        "--load_dataset_name", type=str, default="12.Klein",
        help="(str) Folder that contains all the relevant input files. For example, if your expression "
        "matrix is in /fs/ess/PCON1234/Brutus/datasets/12.Klein/T2000_expression.csv, this should be "
        "set to 12.Klein")
    parser.add_argument(
        "--load_use_benchmark", action="store_true", default=False,
        help="(boolean, default False) If True, expect the following files (replace DATASET_NAME with "
        "the input to the --load_dataset_name argument): `ind.DATASET_NAME.{x, tx, allx}`, "
        "`T2000_expression.csv`, `T2000_LTMG.txt`, `DATASET_NAME_cell_label.csv` if providing "
        "ground-truth cell type labels, and `DATASET_NAME_bulk.csv` if using bulk data")
    parser.add_argument("--load_sc_dataset", type=str, default="", help="Not needed if using benchmark")
    parser.add_argument("--load_bulk_dataset", type=str, default="", help="Not needed if using benchmark")
    parser.add_argument("--load_cell_type_labels", type=str, default="", help="Not needed if using benchmark")
    parser.add_argument("--load_LTMG", type=str, default=None, help="Not needed if using benchmark")

    # Preprocess related
    parser.add_argument("--preprocess_cell_cutoff", type=float, default=0.9, help="Not needed if using benchmark")
    parser.add_argument("--preprocess_gene_cutoff", type=float, default=0.9, help="Not needed if using benchmark")
    parser.add_argument("--preprocess_top_gene_select", type=int, default=2000, help="Not needed if using benchmark")

    # Feature AE related
    parser.add_argument(
        "--feature_AE_epoch", nargs=2, type=int, default=[500, 300],
        help="(two integers separated by a space, default 500 200) First number being non-EM epochs, "
        "second number being EM epochs")
    parser.add_argument("--feature_AE_batch_size", type=int, default=12800, help="(int, default 12800) Batch size")
    parser.add_argument("--feature_AE_learning_rate", type=float, default=1e-3,
                        help="(float, default 1e-3) Learning rate")
    parser.add_argument(
        "--feature_AE_regu_strength", type=float, default=0.9,
        help="(float, default 0.9) In loss function, this is the weight on the LTMG regularization "
        "matrix")
    parser.add_argument("--feature_AE_dropout_prob", type=float, default=0,
                        help="(float, default 0) The dropout probability for feature autoencoder")
    parser.add_argument("--feature_AE_concat_prev_embed", type=str, default=None,
                        help="(str, default None) Choose from {'feature', 'graph'}")

    # Graph AE related
    parser.add_argument("--graph_AE_epoch", type=int, default=200,
                        help="(int, default 200) The epoch or graph autoencoder")
    parser.add_argument(
        "--graph_AE_use_GAT", action="store_true", default=False,
        help="(boolean, default False) If true, will use GAT for GAE layers; otherwise will use GCN "
        "layers")
    parser.add_argument("--graph_AE_GAT_dropout", type=float, default=0,
                        help="(int, default 0) The dropout probability for GAT")
    parser.add_argument("--graph_AE_learning_rate", type=float, default=1e-2,
                        help="(float, default 1e-2) Learning rate")
    parser.add_argument("--graph_AE_embedding_size", type=int, default=16,
                        help="(int, default 16) Graphh AE embedding size")
    parser.add_argument(
        "--graph_AE_concat_prev_embed", action="store_true", default=False,
        help="(boolean, default False) If true, will concat GAE embed at t-1 with the inputted Feature "
        "AE embed at t for graph construction; else will construct graph using Feature AE embed only")
    parser.add_argument("--graph_AE_normalize_embed", type=str, default=None,
                        help="(str, default None) Choose from {None, 'sum1', 'binary'}")
    parser.add_argument("--graph_AE_graph_construction", type=str, default="v2",
                        help="(str, default v0) Choose from {'v0', 'v1', 'v2'}")
    parser.add_argument("--graph_AE_neighborhood_factor", type=float, default=0.05, help="(int, default 10)")
    parser.add_argument("--graph_AE_retain_weights", action="store_true", default=False,
                        help="(boolean, default False)")
    parser.add_argument("--gat_multi_heads", type=int, default=2, help="(int, default 2)")
    parser.add_argument("--gat_hid_embed", type=int, default=64, help="(int, default 64) The dim for hid_embed")

    # Clustering related
    parser.add_argument(
        "--clustering_louvain_only", action="store_true", default=False,
        help="(boolean, default False) If true, will use Louvain clustering only; otherwise, first use "
        "Louvain to determine clusters count (k), then perform KMeans.")
    parser.add_argument(
        "--clustering_use_flexible_k", action="store_true", default=False,
        help="(boolean, default False) If true, will determine k using Louvain every epoch; otherwise, "
        "will rely on the k in the first epoch")
    parser.add_argument("--clustering_embed", type=str, default="graph",
                        help="(str, default 'graph') Choose from {'feature', 'graph', 'both'}")
    parser.add_argument("--clustering_method", type=str, default="KMeans",
                        help="(str, default 'KMeans') Choose from {'KMeans', 'AffinityPropagation'}")

    # Cluster AE related
    parser.add_argument("--cluster_AE_epoch", type=int, default=200, help="(int, default 200) The epoch for cluster AE")
    parser.add_argument("--cluster_AE_batch_size", type=int, default=12800, help="(int, default 12800) Batch size")
    parser.add_argument("--cluster_AE_learning_rate", type=float, default=1e-3,
                        help="(float, default 1e-3) Learning rate")
    parser.add_argument(
        "--cluster_AE_regu_strength", type=float, default=0.9,
        help="(float, default 0.9) In loss function, this is the weight on the LTMG regularization "
        "matrix")
    parser.add_argument("--cluster_AE_dropout_prob", type=float, default=0,
                        help="(float, default 0) The dropout probability for cluster AE")

    # Deconvolution related
    parser.add_argument("--deconv_opt1_learning_rate", type=float, default=1e-3,
                        help="(float, default 1e-3) learning rate")
    parser.add_argument("--deconv_opt1_epoch", type=int, default=5000, help="(int, default 5000) epoch")
    parser.add_argument("--deconv_opt1_epsilon", type=float, default=1e-4, help="(float, default 1e-4) epsilon")
    parser.add_argument("--deconv_opt1_regu_strength", type=float, default=1e-2, help="(float, default 1e-2) strength")

    parser.add_argument("--deconv_opt2_learning_rate", type=float, default=1e-1,
                        help="(float, default 1e-1) learning rate")
    parser.add_argument("--deconv_opt2_epoch", type=int, default=500, help="(int, default 500) epoch")
    parser.add_argument("--deconv_opt2_epsilon", type=float, default=1e-4, help="(float, default 1e-4) epsilon")
    parser.add_argument("--deconv_opt2_regu_strength", type=float, default=1e-2, help="(float, default 1e-2) strength")

    parser.add_argument("--deconv_opt3_learning_rate", type=float, default=1e-1, help="(float, default 1e-1)")
    parser.add_argument("--deconv_opt3_epoch", type=int, default=150, help="(int, default 150) epoch")
    parser.add_argument("--deconv_opt3_epsilon", type=float, default=1e-4, help="(float, default 1e-4) epsilon")
    parser.add_argument("--deconv_opt3_regu_strength_1", type=float, default=0.8,
                        help="(float, default 0.8) strength_1")
    parser.add_argument("--deconv_opt3_regu_strength_2", type=float, default=1e-2,
                        help="(float, default 1e-2) strength_2")
    parser.add_argument("--deconv_opt3_regu_strength_3", type=float, default=1, help="(float, default 1) strength_3")

    parser.add_argument("--deconv_tune_learning_rate", type=float, default=1e-2,
                        help="(float, default 1e-2) learning rate")
    parser.add_argument("--deconv_tune_epoch", type=int, default=20, help="(int, default 20) epoch")
    parser.add_argument("--deconv_tune_epsilon", type=float, default=1e-4, help="(float, default) epsilon")
    parser.add_argument("--data_dir", type=str, default='data', help='test directory')
    parser.add_argument("--dataset", default='mouse_brain_data', type=str, help="dataset id")
    parser.add_argument("--train_size", type=float, default=0.9, help="proportion of training set")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")

    args = parser.parse_args()
    rmses = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)
        logger.info(pformat(vars(args)))

        preprocessing_pipeline = Compose(
            FilterGenesScanpy(min_cells=0.01),
            FilterCellsScanpy(min_genes=0.01),
            # FilterGenesTopK(num_genes=2000, mode="var"),
            CellwiseMaskData(),
            AnnDataTransform(sc.pp.log1p),
            log_level="INFO",
        )
        dataloader = ImputationDataset(data_dir=args.data_dir, dataset=args.dataset, train_size=args.train_size)
        data = dataloader.load_data(transform=preprocessing_pipeline)

        x_train = data.data.X.A * data.data.layers["train_mask"]
        test_mask = data.data.layers["valid_mask"]

        model = ScGNN2(args)

        model.fit(x_train)
        test_mse = ((data.data.X.A[test_mask] - model.predict()[test_mask])**2).mean()
        print(f"MSE: {test_mse:.4f}, RMSE: {np.sqrt(test_mse):.4f}")
        rmses.append(np.sqrt(test_mse))

    print('scgnn')
    print(args.dataset)
    print(f'rmses: {rmses}')
    print(f'rmses: {np.mean(rmses)} +/- {np.std(rmses)}')
"""

Mouse Brain
CUDA_VISIBLE_DEVICES=1 python scgnn2.py --dataset mouse_brain_data --feature_AE_epoch 20 10 --cluster_AE_epoch 20 --total_epoch 2

Mouse Embryo
CUDA_VISIBLE_DEVICES=1 python scgnn2.py --dataset mouse_embryo_data --feature_AE_epoch 20 10 --cluster_AE_epoch 20 --total_epoch 2

PBMC
CUDA_VISIBLE_DEVICES=6 python scgnn2.py --dataset pbmc_data --feature_AE_epoch 20 10 --cluster_AE_epoch 20 --total_epoch 2

"""
