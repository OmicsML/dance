"""Reimplementation of graph-sc.

Extended from https://github.com/ciortanmadalina/graph-sc

Reference
----------
Ciortan, Madalina, and Matthieu Defrance. "GNN-based embedding for clustering scRNA-seq data." Bioinformatics 38.4
(2022) 1037-1044.

"""
import dgl
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLError
from dgl import function as fn
from dgl.nn.pytorch import GraphConv
from dgl.utils import expand_as_pair
from sklearn.cluster import KMeans
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from tqdm import tqdm

from dance import logger
from dance.modules.base import BaseClusteringMethod
from dance.transforms import AnnDataTransform, Compose, SetConfig
from dance.transforms.graph import PCACellFeatureGraph
from dance.typing import Any, Literal, LogLevel, Optional
from dance.utils import get_device


class GraphSC(BaseClusteringMethod):
    """GraphSC class.

    Parameters
    ----------
    agg
        Aggregation layer.
    activation
        Activation function.
    in_feats
        Dimension of input feature
    n_hidden
        Number of hidden layer.
    hidden_dim
        Input dimension of hidden layer 1.
    hidden_1
        Output dimension of hidden layer 1.
    hidden_2
        Output dimension of hidden layer 2.
    dropout
        Dropout rate.
    n_layers
        Number of graph convolutional layers.
    hidden_relu
        Use relu activation in hidden layers or not.
    hidden_bn
        Use batch norm in hidden layers or not.
    cluster_method
        Method for clustering.
    num_workers
        Number of workers.
    device
        Computation device to use.

    """

    def __init__(
        self,
        agg: str = "sum",
        activation: str = "relu",
        in_feats: int = 50,
        n_hidden: int = 1,
        hidden_dim: int = 200,
        hidden_1: int = 300,
        hidden_2: int = 0,
        dropout: float = 0.1,
        n_layers: int = 1,
        hidden_relu: bool = False,
        hidden_bn: bool = False,
        n_clusters: int = 10,
        cluster_method: Literal["kmeans", "leiden"] = "kmeans",
        num_workers: int = 1,
        device: str = "auto",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self.num_workers = num_workers
        self.device = get_device(device)

        self.model = GCNAE(
            agg=agg,
            activation=activation,
            in_feats=in_feats,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            dropout=dropout,
            n_layers=n_layers,
            hidden_relu=hidden_relu,
            hidden_bn=hidden_bn,
        ).to(self.device)

    @staticmethod
    def preprocessing_pipeline(n_top_genes: int = 3000, normalize_weights: str = "log_per_cell", n_components: int = 50,
                               normalize_edges: bool = False, log_level: LogLevel = "INFO"):
        transforms = [
            AnnDataTransform(sc.pp.filter_genes, min_counts=3),
            AnnDataTransform(sc.pp.filter_cells, min_counts=1),
            AnnDataTransform(sc.pp.normalize_total),
            AnnDataTransform(sc.pp.log1p),
            AnnDataTransform(sc.pp.highly_variable_genes, min_mean=0.0125, max_mean=4, flavor="cell_ranger",
                             min_disp=0.5, n_top_genes=n_top_genes, subset=True),
        ]

        if normalize_weights == "log_per_cell":
            transforms.extend([
                AnnDataTransform(sc.pp.log1p),
                AnnDataTransform(sc.pp.normalize_total, target_sum=1),
            ])
        elif normalize_weights == "per_cell":
            transforms.append(AnnDataTransform(sc.pp.normalize_total, target_sum=1))
        elif normalize_weights != "none":
            raise ValueError(f"Unknown normalization option {normalize_weights!r}."
                             "Available options are: 'none', 'log_per_cell', 'per_cell'")

        # Cell-gene graph construction
        transforms.extend([
            PCACellFeatureGraph(
                n_components=n_components,
                normalize_edges=normalize_edges,
                feat_norm_mode="standardize",
            ),
            SetConfig({
                "feature_channel": "CellFeatureGraph",
                "feature_channel_type": "uns",
                "label_channel": "Group",
            }),
        ])

        return Compose(*transforms, log_level=log_level)

    def fit(
        self,
        g: dgl.DGLGraph,
        y: Optional[Any] = None,
        *,
        epochs: int = 100,
        lr: float = 1e-5,
        batch_size: int = 128,
        show_epoch_ari: bool = False,
        eval_epoch: bool = False,
    ):
        """Train graph-sc.

        Parameters
        ----------
        g
            Input cell-gene graph.
        y
            Not used, for compatibility with the BaseClusteringMethod class.
        epochs
            Number of epochs.
        lr
            Learning rate.
        batch_size
            Batch size.
        show_epoch_ari
            Show ARI score for each epoch
        eval_epoch
            Evaluate every epoch.

        """
        g.ndata["order"] = g.ndata["label"] = g.ndata["feat_id"]
        train_ids = np.where(g.ndata["label"] != -1)[0]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        dataloader = dgl.dataloading.DataLoader(g, train_ids, sampler, batch_size=batch_size, shuffle=True,
                                                drop_last=False, num_workers=self.num_workers)

        device = get_device(self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        aris = []
        Z = {}

        for epoch in tqdm(range(epochs)):
            self.model.train()
            z = []
            y = []
            order = []

            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata["features"]
                g = blocks[-1]

                adj_logits, emb = self.model.forward(blocks, input_features)
                z.extend(emb.detach().cpu().numpy())
                if "label" in blocks[-1].dstdata:
                    y.extend(blocks[-1].dstdata["label"].cpu().numpy())
                order.extend(blocks[-1].dstdata["order"].cpu().numpy())

                adj = g.adjacency_matrix().to_dense().to(device)
                adj = adj[g.dstnodes()][:, g.dstnodes()]
                pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
                factor = float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
                if factor == 0:
                    factor = 1
                norm = adj.shape[0] * adj.shape[0] / factor
                adj_logits, _ = self.model.forward(blocks, input_features)
                loss = norm * BCELoss(adj_logits, adj.to(device), pos_weight=pos_weight.to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())

            z = np.array(z)
            y = np.array(y)
            order = np.array(order)
            order = np.argsort(order)
            z = z[order]
            y = y[order]
            if pd.isnull(y[0]):
                y = None
            self.z = z

            if eval_epoch:
                score = self.score(None, y)
                aris.append(score)
                if show_epoch_ari:
                    logger.info(f"epoch {epoch:4d}, ARI {score:.4f}")
                z_ = {f"epoch{epoch}": z}
                Z = {**Z, **z_}

            elif epoch == epochs - 1:
                self.z = z

        if eval_epoch:
            index = np.argmax(aris)
            self.z = Z[f"epoch{index}"]

    def predict(self, x: Optional[Any] = None):
        """Get predictions from the graph autoencoder model.

        Parameters
        ----------
        x
            Not used, for compatibility with BaseClusteringMethod class.

        Returns
        -------
        pred
            Prediction of given clustering method.

        """
        if self.cluster_method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", random_state=5, n_init=10)
            pred = kmeans.fit_predict(self.z)
        elif self.cluster_method == "leiden":
            pred = run_leiden(self.z)
        else:
            raise ValueError(f"Unknown clustering {self.cluster_method}, available options are: 'kmeans', 'leiden'")
        return pred


class GCNAE(nn.Module):
    """Graph convolutional autoencoder class.

    Parameters
    ----------
    agg
        Aggregation layer.
    activation
        Activation function.
    in_feats
        Dimension of input feature
    n_hidden
        Number of hidden layer.
    hidden_dim
        Input dimension of hidden layer 1.
    hidden_1
        Output dimension of hidden layer 1.
    hidden_2
        Output dimension of hidden layer 2.
    dropout
        Dropout rate.
    n_layers
        Number of graph convolutional layers.
    hidden_relu
        Use relu activation in hidden layers or not.
    hidden_bn
        Use batch norm in hidden layers or not.

    Returns
    -------
    adj_rec
        Reconstructed adjacency matrix.
    x
        Embedding.

    """

    def __init__(
        self,
        *,
        agg: str,
        activation: str,
        in_feats: int,
        n_hidden: int,
        hidden_dim: int,
        hidden_1: int,
        hidden_2: int,
        dropout: float,
        n_layers: int,
        hidden_relu: bool,
        hidden_bn: bool,
    ):
        super().__init__()

        self.agg = agg

        if activation == "gelu":
            activation = F.gelu
        elif activation == "prelu":
            activation = F.prelu
        elif activation == "relu":
            activation = F.relu
        elif activation == "leaky_relu":
            activation = F.leaky_relu

        if n_hidden == 0:
            hidden = None
        elif n_hidden == 1:
            hidden = [hidden_1]
        elif n_hidden == 2:
            hidden = [hidden_1, hidden_2]

        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.layer1 = WeightedGraphConv(in_feats=in_feats, out_feats=hidden_dim, activation=activation)
        if n_layers == 2:
            self.layer2 = WeightedGraphConv(in_feats=hidden_dim, out_feats=hidden_dim, activation=activation)
        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.hidden = hidden
        if hidden is not None:
            enc = []
            for i, s in enumerate(hidden):
                if i == 0:
                    enc.append(nn.Linear(hidden_dim, hidden[i]))
                else:
                    enc.append(nn.Linear(hidden[i - 1], hidden[i]))
                if hidden_bn and i != len(hidden):
                    enc.append(nn.BatchNorm1d(hidden[i]))
                if hidden_relu and i != len(hidden):
                    enc.append(nn.ReLU())
            self.encoder = nn.Sequential(*enc)

    def forward(self, blocks, features):
        x = blocks[0].srcdata["features"]
        for i in range(len(blocks)):
            with blocks[i].local_scope():
                if self.dropout is not None:
                    x = self.dropout(x)
                blocks[i].srcdata["h"] = x
                if i == 0:
                    x = self.layer1(blocks[i], x, agg=self.agg)
                else:
                    x = self.layer2(blocks[i], x, agg=self.agg)
        if self.hidden is not None:
            x = self.encoder(x)
        adj_rec = self.decoder(x)
        return adj_rec, x


class InnerProductDecoder(nn.Module):
    """Inner product decoder class.

    Parameters
    ----------
    activation
        Activation function.
    dropout
        Dropout rate.

    Returns
    -------
    adj
        Reconstructed adjacency matrix.

    """

    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj


class WeightedGraphConv(GraphConv):
    """Adaptation of the dgl GraphConv model to use edge weights."""

    def edge_selection_simple(self, edges):
        """Edge selection.

        Parameters
        ----------
        edges
            Edges of graph.

        """
        return {"m": edges.src["h"] * edges.data["weight"]}

    def forward(self, graph, feat, weight=None, agg="sum"):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError("There are 0-in-degree nodes in the graph, "
                                   "output for those nodes will be invalid. "
                                   "This is harmful for some applications, "
                                   "causing silent performance regression. "
                                   "Adding self-loop on the input graph by "
                                   "calling `g = dgl.add_self_loop(g)` will resolve "
                                   "the issue. Setting ``allow_zero_in_degree`` "
                                   "to be `True` when constructing this module will "
                                   "suppress the check and let the code run.")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == "both":
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1, ) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError("External weight is provided while at the same time the"
                                   " module has defined its own weight parameter. Please"
                                   " create the module with flag weight=False.")
            else:
                weight = self.weight

            if weight is not None:
                feat_src = torch.matmul(feat_src, weight)
            graph.srcdata["h"] = feat_src
            if agg == "sum":
                graph.update_all(self.edge_selection_simple, fn.sum(msg="m", out="h"))
            if agg == "mean":
                graph.update_all(self.edge_selection_simple, fn.mean(msg="m", out="h"))
            rst = graph.dstdata["h"]
            if self._norm != "none":

                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1, ) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class WeightedGraphConvAlpha(GraphConv):
    """Adaptation of the dgl GraphConv model to learn the extra edge weight
    parameter."""

    def edge_selection_simple(self, edges):
        """Edge selection.

        Parameters
        ----------
        edges
            Edges of graph.

        """
        number_of_edges = edges.src["h"].shape[0]
        indices = np.expand_dims(np.array([self.gene_num + 1] * number_of_edges, dtype=np.int32), axis=1)
        src_id, dst_id = edges.src["id"].cpu().numpy(), edges.dst["id"].cpu().numpy()
        indices = np.where((src_id >= 0) & (dst_id < 0), src_id, indices)  # gene->cell
        indices = np.where((dst_id >= 0) & (src_id < 0), dst_id, indices)  # cell->gene
        indices = np.where((dst_id >= 0) & (src_id >= 0), self.gene_num, indices)  # gene-gene
        h = edges.src["h"] * self.alpha[indices.squeeze()]
        return {"m": h}

    def forward(self, graph, feat, weight=None, alpha=None, gene_num=None):
        self.alpha = alpha
        self.gene_num = gene_num
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError("There are 0-in-degree nodes in the graph, "
                                   "output for those nodes will be invalid. "
                                   "This is harmful for some applications, "
                                   "causing silent performance regression. "
                                   "Adding self-loop on the input graph by "
                                   "calling `g = dgl.add_self_loop(g)` will resolve "
                                   "the issue. Setting ``allow_zero_in_degree`` "
                                   "to be `True` when constructing this module will "
                                   "suppress the check and let the code run.")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == "both":
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1, ) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError("External weight is provided while at the same time the"
                                   " module has defined its own weight parameter. Please"
                                   " create the module with flag weight=False.")
                else:
                    feat_src = torch.matmul(feat_src, weight)
            else:
                weight = self.weight

            graph.srcdata["h"] = feat_src
            graph.update_all(self.edge_selection_simple, fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]

            if self._norm != "none":

                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1, ) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


def run_leiden(data):
    """Performs Leiden community detection on given data.

    Parameters
    ----------
    data
        Aata for leiden

    Returns
    -------
    pred
        Prediction of leiden.

    """
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=300, n_pcs=0)
    sc.tl.leiden(adata)
    pred = adata.obs["leiden"].to_list()
    pred = [int(x) for x in pred]
    return pred
