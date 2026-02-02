import argparse
import pprint
from typing import Optional, Union, get_args

import dgl
import numpy as np
from sklearn.decomposition import PCA
import torch

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
@register_preprocessor("feature", "cell",overwrite=True)
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
            self.logger.info(f"Normalizing feature before PCA decomposition with mode={self.feat_norm_mode} "
                             f"and axis={self.feat_norm_axis}")
            feat = normalize(feat, mode=self.feat_norm_mode, axis=self.feat_norm_axis)
        if self.n_components > min(feat.shape):
            self.logger.warning(
                f"n_components={self.n_components} must be between 0 and min(n_samples, n_features)={min(feat.shape)} with svd_solver='full'"
            )
            self.n_components = min(feat.shape)
        gene_pca = PCA(n_components=self.n_components)  # genes x components

        gene_feat = gene_pca.fit_transform(feat.T)  # decompose into gene features
        # self.logger.info(f"Decomposing {self.split_name} features {feat.shape} (k={gene_pca.n_components_})")
        # self.logger.info(f"Total explained variance: {gene_pca.explained_variance_ratio_.sum():.2%}")

        x = data.get_x()
        cell_feat = normalize(x, mode="normalize", axis=1) @ gene_feat  # cells x components
        data.data.obsm[self.out] = cell_feat.astype(np.float32)
        data.data.varm[self.out] = gene_feat.astype(np.float32)
        # if self.save_info:
        #     data.data.uns["pca_components"] = gene_pca.components_
        #     data.data.uns["pca_mean"] = gene_pca.mean_
        #     data.data.uns["pca_explained_variance"] = gene_pca.explained_variance_
        #     data.data.uns["pca_explained_variance_ratio"] = gene_pca.explained_variance_ratio_
        return data

@register_preprocessor("graph", "cell",overwrite=True)
class CellFeatureGraph(BaseTransform):

    def __init__(self, cell_feature_channel: str, gene_feature_channel: Optional[str] = None, *,
                 mod: Optional[str] = None, normalize_edges: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.cell_feature_channel = cell_feature_channel
        self.gene_feature_channel = gene_feature_channel or cell_feature_channel
        self.mod = mod
        self.normalize_edges = normalize_edges

    def _alternative_construction(self, data):
        # TODO: Try this alternative construction
        x_sparse = data.get_feature(return_type="sparse", channel=self.cell_feature_channel, mod=self.mod)
        g = dgl.bipartite_from_scipy(x_sparse, utype="cell", etype="expression", vtype="feature", eweight_name="weight")
        g = dgl.ToSimple()(g)
        g = dgl.AddSelfLoop(edge_feat_names="weight")(g)
        g = dgl.AddReverse(copy_edata=True)(g)
        g.ndata["weight"] = dgl.nn.EdgeWeightNorm(norm="both")(g, g.ndata["weight"])
        data.data.uns[self.out] = g
        return data

    def __call__(self, data):
        feat = data.get_feature(return_type="default", mod=self.mod)
        num_cells, num_feats = feat.shape

        row, col = np.nonzero(feat)
        edata = np.array(feat[row, col]).ravel()[:, None]
        self.logger.info(f"Number of nonzero entries: {edata.size:,}")
        self.logger.info(f"Nonzero rate = {edata.size / num_cells / num_feats:.1%}")

        row = row + num_feats  # offset by feature nodes
        col, row = np.hstack((col, row)), np.hstack((row, col))  # convert to undirected
        edata = np.vstack((edata, edata))

        # Convert to tensors
        col = torch.LongTensor(col)
        row = torch.LongTensor(row)
        edata = torch.FloatTensor(edata)

        # Initialize cell-gene graph
        g = dgl.graph((row, col))
        g.edata["weight"] = edata
        # FIX: change to feat_id
        g.ndata["cell_id"] = torch.concat((torch.arange(num_feats, dtype=torch.int32),
                                           -torch.ones(num_cells, dtype=torch.int32)))  # yapf: disable
        g.ndata["feat_id"] = torch.concat((-torch.ones(num_feats, dtype=torch.int32),
                                           torch.arange(num_cells, dtype=torch.int32)))  # yapf: disable

        # Normalize edges and add self-loop
        if self.normalize_edges:
            in_deg = g.in_degrees()
            for i in range(g.number_of_nodes()):
                src, dst, eidx = g.in_edges(i, form="all")
                if src.shape[0] > 0:
                    edge_w = g.edata["weight"][eidx]
                    g.edata["weight"][eidx] = in_deg[i] * edge_w / edge_w.sum()
        g.add_edges(g.nodes(), g.nodes(), {"weight": torch.ones(g.number_of_nodes())[:, None]})

        gene_feature = data.get_feature(return_type="torch", channel=self.gene_feature_channel, mod=self.mod,
                                        channel_type="varm")
        cell_feature = data.get_feature(return_type="torch", channel=self.cell_feature_channel, mod=self.mod,
                                        channel_type="obsm")
        g.ndata["features"] = torch.vstack((gene_feature, cell_feature))

        data.data.uns[self.out] = g

        return data


@register_preprocessor("graph", "cell",overwrite=True)
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
                                               train_dataset=args.train_dataset, data_dir="../temp_data", val_size=args.val_size)
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
