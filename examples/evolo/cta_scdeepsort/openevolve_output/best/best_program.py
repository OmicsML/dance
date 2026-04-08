import argparse
import pprint
from typing import Optional, Union, get_args

import dgl
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed
from dance.utils.matrix import normalize
from dance.utils.wrappers import add_mod_and_transform


# EVOLVE-BLOCK-START
@register_preprocessor("feature", "cell", overwrite=True)
@add_mod_and_transform
class WeightedFeaturePCA(BaseTransform):
    """Compute the weighted gene PCA as cell features.

    Given a gene expression matrix of dimension (cell x gene), the gene PCA is first compured. Then, the representation
    of each cell is computed by taking the weighted sum of the gene PCAs based on that cell's gene expression values.

    Parameters
    ----------
    n_components
        Number of PCs to use.
    split_name
        Which split to use to compute the gene PCA. If not set, use all data.
    feat_norm_mode
        Feature normalization mode, see :func:`dance.utils.matrix.normalize`. If set to `None`, then do not perform
        feature normalization before reduction.

    """

    _DISPLAY_ATTRS = ("n_components", "split_name", "feat_norm_mode", "feat_norm_axis")

    def __init__(self, n_components: Union[float, int] = 400, split_name: Optional[str] = None,
                 feat_norm_mode: Optional[str] = None, feat_norm_axis: int = 0, save_info=False, **kwargs):
        super().__init__(**kwargs)

        self.n_components = n_components
        self.split_name = split_name
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis
        self.save_info = save_info

    def __call__(self, data):
        feat = data.get_x(self.split_name)  # cell x genes
        if self.feat_norm_mode is not None:
            self.logger.info(f"Normalizing feature before decomposition with mode={self.feat_norm_mode} "
                             f"and axis={self.feat_norm_axis}")
            feat = normalize(feat, mode=self.feat_norm_mode, axis=self.feat_norm_axis)

        # Use TruncatedSVD instead of PCA for sparse matrices
        if self.n_components > min(feat.shape):
            self.logger.warning(
                f"n_components={self.n_components} must be between 0 and min(n_samples, n_features)={min(feat.shape)}")
            self.n_components = min(feat.shape) - 1  # TruncatedSVD requires n_components < min(shape)

        # Apply log transformation to stabilize variance
        feat_log = np.log1p(feat)

        gene_decomposer = TruncatedSVD(n_components=self.n_components)  # genes x components

        gene_feat = gene_decomposer.fit_transform(feat_log.T)  # decompose into gene features using log-transformed data

        # Compute cell features independently using direct TruncatedSVD on cell-by-gene matrix
        cell_decomposer = TruncatedSVD(n_components=self.n_components)
        cell_feat_direct = cell_decomposer.fit_transform(feat_log)  # cells x components using log-transformed data

        x = data.get_x()
        # Use log-transformed normalized expression values for weighting
        x_log = np.log1p(x)  # log(1+x) transformation
        x_norm = normalize(x_log, mode="normalize", axis=1)
        cell_feat_weighted = x_norm @ gene_feat  # cells x components

        # Combine both representations with optimized weights
        cell_feat = 0.6 * cell_feat_weighted + 0.4 * cell_feat_direct

        data.data.obsm[self.out] = cell_feat.astype(np.float32)
        data.data.varm[self.out] = gene_feat.astype(np.float32)
        return data


@register_preprocessor("graph", "cell", overwrite=True)
class CellFeatureGraph(BaseTransform):

    def __init__(self, cell_feature_channel: str, gene_feature_channel: Optional[str] = None, *,
                 mod: Optional[str] = None, normalize_edges: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.cell_feature_channel = cell_feature_channel
        self.gene_feature_channel = gene_feature_channel or cell_feature_channel
        self.mod = mod
        self.normalize_edges = normalize_edges

    def __call__(self, data):
        feat = data.get_feature(return_type="default", mod=self.mod)
        num_cells, num_genes = feat.shape

        # Apply enhanced TF-IDF transformation to highlight marker genes
        # First apply log transformation
        feat_log = np.log1p(feat)

        # Calculate TF-IDF manually for more control
        tf_matrix = feat_log  # Term frequency after log transformation

        # Calculate document frequency (how many cells express each gene)
        df = np.sum(feat > 0, axis=0)  # Count non-zero expressions per gene
        total_cells = feat.shape[0]
        idf = np.log(total_cells / (df + 1))  # Inverse document frequency with smoothing

        # Calculate TF-IDF
        feat_tfidf = tf_matrix * idf  # Broadcasting multiplication

        # Find top-k connections per cell to sparsify the graph
        # This reduces noise and focuses on important gene-cell relationships
        k = min(50, num_genes // 2)  # Use top 50 or half the genes, whichever is smaller

        # For each cell, find top k expressed genes
        top_k_indices = np.argpartition(feat_tfidf, -k, axis=1)[:, -k:]
        row_indices = np.repeat(np.arange(num_cells), k)
        col_indices = top_k_indices.flatten()

        # Get corresponding weights
        weights = feat_tfidf[row_indices, col_indices]

        # Filter out zero weights to ensure sparse representation
        mask = weights > 0
        row_indices = row_indices[mask]
        col_indices = col_indices[mask]
        weights = weights[mask]

        self.logger.info(f"Number of nonzero entries after sparsification: {len(weights):,}")
        self.logger.info(f"Sparsity rate = {len(weights) / num_cells / num_genes:.1%}")

        # Offset gene indices to distinguish from cell nodes
        gene_node_indices = col_indices + num_cells  # gene nodes start after cell nodes

        # Create bidirectional edges (cell->gene and gene->cell)
        row = np.concatenate([row_indices, gene_node_indices])
        col = np.concatenate([gene_node_indices, row_indices])
        edata = np.concatenate([weights, weights])[:, None]

        # Convert to tensors
        row = torch.LongTensor(row)
        col = torch.LongTensor(col)
        edata = torch.FloatTensor(edata)

        # Initialize bipartite cell-gene graph
        g = dgl.graph((row, col))
        g.edata["weight"] = edata

        # Store node type information
        g.ndata["node_type"] = torch.cat([
            torch.zeros(num_cells, dtype=torch.int32),  # 0 for cells
            torch.ones(num_genes, dtype=torch.int32)  # 1 for genes
        ])

        # Apply efficient edge normalization using DGL's built-in function
        if self.normalize_edges:
            edge_norm = dgl.nn.EdgeWeightNorm(norm='both')
            g.edata['weight'] = edge_norm(g, g.edata['weight'])

        gene_feature = data.get_feature(return_type="torch", channel=self.gene_feature_channel, mod=self.mod,
                                        channel_type="varm")
        cell_feature = data.get_feature(return_type="torch", channel=self.cell_feature_channel, mod=self.mod,
                                        channel_type="obsm")

        # Concatenate cell and gene features
        g.ndata["features"] = torch.vstack((cell_feature, gene_feature))

        data.data.uns[self.out] = g

        return data


@register_preprocessor("graph", "cell", overwrite=True)
class PCACellFeatureGraph(BaseTransform):

    _DISPLAY_ATTRS = ("n_components", "split_name")

    def __init__(
        self,
        n_components: int = 400,
        split_name: Optional[str] = None,
        *,
        normalize_edges: bool = True,
        feat_norm_mode: Optional[str] = None,
        feat_norm_axis: int = 0,
        mod: Optional[str] = None,
        log_level: LogLevel = "WARNING",
    ):
        super().__init__(log_level=log_level)

        self.n_components = n_components
        self.split_name = split_name
        self.normalize_edges = normalize_edges
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis
        self.mod = mod

    def __call__(self, data):
        WeightedFeaturePCA(self.n_components, self.split_name, feat_norm_mode=self.feat_norm_mode,
                           feat_norm_axis=self.feat_norm_axis, log_level=self.log_level)(data)
        CellFeatureGraph(cell_feature_channel="WeightedFeaturePCA", mod=self.mod, normalize_edges=self.normalize_edges,
                         log_level=self.log_level)(data)
        return data


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dense_dim", type=int, default=400, help="number of hidden gcn units")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--hidden_dim", type=int, default=200, help="number of hidden gcn units")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--val_size", type=float, default=0.0, help="val size")
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"Running SVM with the following parameters:\n{pprint.pformat(vars(args))}")

    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = ScDeepSort(args.dense_dim, args.hidden_dim, args.n_layers, args.species, args.tissue,
                           dropout=args.dropout, batch_size=args.batch_size, device=args.device)
        preprocessing_pipeline = Compose(
            PCACellFeatureGraph(n_components=args.dense_dim, split_name="train"),
            SetConfig({"label_channel": "cell_type"}),
            log_level="INFO",
        )

        # Load data and perform necessary preprocessing
        dataloader = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                               train_dataset=args.train_dataset, data_dir="../temp_data",
                                               val_size=args.val_size)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        # Obtain training and testing data
        y_train = data.get_y(split_name="train", return_type="torch")
        y_test = data.get_y(split_name="test", return_type="torch")
        num_labels = y_test.shape[1]

        # Get cell feature graph for scDeepSort
        # TODO: make api for the following block?
        g = data.data.uns["CellFeatureGraph"]
        num_genes = data.shape[1]
        gene_ids = torch.arange(num_genes)
        train_cell_ids = torch.LongTensor(data.train_idx) + num_genes
        test_cell_ids = torch.LongTensor(data.test_idx) + num_genes
        g_train = g.subgraph(torch.concat((gene_ids, train_cell_ids)))
        g_test = g.subgraph(torch.concat((gene_ids, test_cell_ids)))

        # Train and evaluate the model
        model.fit(g_train, y_train.argmax(1), epochs=args.n_epochs, lr=args.lr, weight_decay=args.weight_decay,
                  val_ratio=args.test_rate)
        score = model.score(g_test, y_test)
        inner_score = model.score(g_train, y_train)
        scores.append(score.item())
        inner_scores.append(inner_score.item())
        print(f"{score=:.4f}")
    print(f"scDeepSort {args.species} {args.tissue} {args.test_dataset}:")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_inner_score = np.mean(inner_scores)
    std_inner_score = np.std(inner_scores)
    print(f"mean_score: {mean_score:.5f} +/- {std_score:.5f}")
    print(f"mean_inner_score: {mean_inner_score:.5f} +/- {std_inner_score:.5f}")
"""To reproduce the benchmarking results, please run the following command:

Mouse Brain
$ python scdeepsort.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python scdeepsort.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python scdeepsort.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
