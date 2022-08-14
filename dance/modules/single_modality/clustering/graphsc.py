"""Reimplementation of graph-sc.

Extended from https://github.com/ciortanmadalina/graph-sc

Reference
----------
Ciortan, Madalina, and Matthieu Defrance. "GNN-based embedding for clustering scRNA-seq data." Bioinformatics 38.4
(2022) 1037-1044.

"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLError, DGLGraph
from dgl import function as fn
from dgl.nn.pytorch import GraphConv
from dgl.utils import expand_as_pair
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, normalized_mutual_info_score, silhouette_score
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from tqdm import tqdm


class GraphSC:
    """GraphSC class.

    Parameters
    ----------
    args : argparse.Namespace
        a Namespace contains arguments of GCNAE. For details of parameters in parser args, please refer to link (parser help document).

    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = GCNAE(args).to(get_device(args.use_cpu))

    def fit(self, n_epochs, dataloader, n_clusters, lr, cluster=["KMeans"]):
        """Train graph-sc.

        Parameters
        ----------
        n_epochs : int
            number of epochs.
        dataloader :
            dataloader for training.
        n_clusters : int
            number of clusters.
        lr : float
            learning rate.
        cluster : list optional
            clustering method.

        Returns
        -------
        None.

        """
        device = get_device(self.args.use_cpu)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        aris_kmeans = []
        Z = {}
        for epoch in tqdm(range(n_epochs)):
            self.model.train()
            z = []
            y = []
            order = []

            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['features']
                g = blocks[-1]
                degs = g.in_degrees().float()

                adj_logits, emb = self.model.forward(blocks, input_features)
                z.extend(emb.detach().cpu().numpy())
                if "label" in blocks[-1].dstdata:
                    y.extend(blocks[-1].dstdata["label"].cpu().numpy())
                order.extend(blocks[-1].dstdata["order"].cpu().numpy())

                adj = g.adjacency_matrix().to_dense()
                adj = adj[g.dstnodes()]
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

            if self.args.eval_epoch:
                score = self.score(y, n_clusters, cluster=cluster)
                aris_kmeans.append(score["kmeans_ari"])
                if self.args.show_epoch_ari:
                    print(f'epoch {epoch}, ARI {score.get("kmeans_ari")}')
                z_ = {f'epoch{epoch}': z}
                Z = {**Z, **z_}

            elif epoch == n_epochs - 1:
                self.z = z

        if self.args.eval_epoch:
            index = np.argmax(aris_kmeans)
            self.z = Z[f'epoch{index}']

    def predict(self, n_clusters, cluster=["KMeans"]):
        """Get predictions from the graph autoencoder model.

        Parameters
        ----------
        n_clusters : int
            number of clusters.
        cluster : list optional
            clustering method.

        Returns
        -------
        pred : dict
            prediction of given clustering method.

        """
        z = self.z
        device = get_device(self.args.use_cpu)
        pred = {}

        if "KMeans" in cluster:
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=5)
            kmeans_pred = {"kmeans_pred": kmeans.fit_predict(z)}
            pred = {**pred, **kmeans_pred}

        if "Leiden" in cluster:
            leiden_pred = {"leiden_pred": run_leiden(z)}
            pred = {**pred, **leiden_pred}

        return pred

    def score(self, y, n_clusters, plot=False, cluster=["KMeans"]):
        """Evaluate the graph autoencoder model.

        Parameters
        ----------
        y : list
            true labels.
        n_clusters : int
            number of clusters.
        plot : bool optional
            show plot or not.
        cluster : list optional
            clustering method.

        Returns
        -------
        scores : dict
            metric evaluation scores.

        """
        z = self.z
        device = get_device(self.args.use_cpu)
        self.model.eval()

        k_start = time.time()
        scores = {"ae_end": k_start}

        if "KMeans" in cluster:
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=5)
            kmeans_pred = kmeans.fit_predict(z)
            ari_k = None
            nmi_k = None
            if y is not None:
                ari_k = round(adjusted_rand_score(y, kmeans_pred), 4)
                nmi_k = round(normalized_mutual_info_score(y, kmeans_pred), 4)
            sil_k = silhouette_score(z, kmeans_pred)
            cal_k = calinski_harabasz_score(z, kmeans_pred)
            k_end = time.time()
            scores_k = {
                "kmeans_ari": ari_k,
                "kmeans_nmi": nmi_k,
                "kmeans_sil": sil_k,
                "kmeans_cal": cal_k,
                "kmeans_pred": kmeans_pred,
                "kmeans_time": k_end - k_start,
            }
            scores = {**scores, **scores_k}

        if "Leiden" in cluster:
            l_start = time.time()
            leiden_pred = run_leiden(z)
            ari_l = None
            nmi_l = None
            if y is not None:
                ari_l = round(adjusted_rand_score(y, leiden_pred), 4)
                nmi_l = round(normalized_mutual_info_score(y, leiden_pred), 4)
            sil_l = silhouette_score(z, leiden_pred)
            cal_l = calinski_harabasz_score(z, leiden_pred)
            l_end = time.time()
            scores_l = {
                "leiden_ari": ari_l,
                "leiden_nmi": nmi_l,
                "leiden_sil": sil_l,
                "leiden_cal": cal_l,
                "leiden_pred": leiden_pred,
                "leiden_time": l_end - l_start,
            }
            scores = {**scores, **scores_l}

        if plot:
            pca = PCA(2).fit_transform(z)
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.title("Ground truth")
            plt.scatter(pca[:, 0], pca[:, 1], c=y, s=4)

            plt.subplot(132)
            plt.title("K-Means pred")
            plt.scatter(pca[:, 0], pca[:, 1], c=kmeans_pred, s=4)

            plt.subplot(133)
            plt.title("Leiden pred")
            plt.scatter(pca[:, 0], pca[:, 1], c=leiden_pred, s=4)
            plt.show()
        return scores


class GCNAE(nn.Module):
    """Graph convolutional autoencoder class.

    Parameters
    ----------
    args : argparse.Namespace
        a Namespace contains arguments of scDSC. For details of parameters in parser args, please refer to link (parser help document).
    agg : str
        aggregation layer.
    activation :str
        activation function.
    in_feats : int
        dimension of input feature
    n_hidden : int
        number of hidden layer.
    hidden_dim :int
        input dimension of hidden layer 1.
    hidden_1 : int
        output dimension of hidden layer 1.
    hidden_2 : int
        output dimension of hidden layer 2.
    dropout : float
        dropout rate.
    n_layers :int
        number of graph convolutional layers.
    hidden_relu : bool
        use relu activation in hidden layers or not.
    hidden_bn : bool
        use batch norm in hidden layers or not.

    Returns
    -------
    adj_rec :
        reconstructed adjacency matrix.
    x :
        embedding.

    """

    def __init__(self, args):

        super().__init__()
        self.args = args
        self.agg = args.agg

        if args.activation == 'gelu':
            activation = F.gelu
        elif args.activation == 'prelu':
            activation = F.prelu
        elif args.activation == 'relu':
            activation = F.relu
        elif args.activation == 'leaky_relu':
            activation = F.leaky_relu

        if args.n_hidden == 0:
            hidden = None
        elif args.n_hidden == 1:
            hidden = [args.hidden_1]
        elif args.n_hidden == 2:
            hidden = [args.hidden_1, args.hidden_2]

        if args.dropout != 0:
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            self.dropout = None

        self.layer1 = WeightedGraphConv(in_feats=args.in_feats, out_feats=args.hidden_dim, activation=activation)
        if args.n_layers == 2:
            self.layer2 = WeightedGraphConv(in_feats=args.hidden_dim, out_feats=args.hidden_dim, activation=activation)
        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.hidden = hidden
        if hidden is not None:
            enc = []
            for i, s in enumerate(hidden):
                if i == 0:
                    enc.append(nn.Linear(args.hidden_dim, hidden[i]))
                else:
                    enc.append(nn.Linear(hidden[i - 1], hidden[i]))
                if args.hidden_bn and i != len(hidden):
                    enc.append(nn.BatchNorm1d(hidden[i]))
                if args.hidden_relu and i != len(hidden):
                    enc.append(nn.ReLU())
            self.encoder = nn.Sequential(*enc)

    def forward(self, blocks, features):
        x = blocks[0].srcdata['features']
        for i in range(len(blocks)):
            with blocks[i].local_scope():
                if self.dropout is not None:
                    x = self.dropout(x)
                blocks[i].srcdata['h'] = x
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
    activation : optional
        activation function.
    dropout : float optional
        dropout rate.

    Returns
    -------
    adj :
        reconstructed adjacency matrix.

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
        edges :
            edges of graph.

        """
        return {'m': edges.src['h'] * edges.data['weight']}

    def forward(self, graph, feat, weight=None, agg="sum"):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1, ) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat_src = torch.matmul(feat_src, weight)
            graph.srcdata['h'] = feat_src
            if agg == "sum":
                graph.update_all(self.edge_selection_simple, fn.sum(msg='m', out='h'))
            if agg == "mean":
                graph.update_all(self.edge_selection_simple, fn.mean(msg='m', out='h'))
            rst = graph.dstdata['h']
            if self._norm != 'none':

                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
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
        edges :
            edges of graph.

        """
        number_of_edges = edges.src['h'].shape[0]
        indices = np.expand_dims(np.array([self.gene_num + 1] * number_of_edges, dtype=np.int32), axis=1)
        src_id, dst_id = edges.src['id'].cpu().numpy(), edges.dst['id'].cpu().numpy()
        indices = np.where((src_id >= 0) & (dst_id < 0), src_id, indices)  # gene->cell
        indices = np.where((dst_id >= 0) & (src_id < 0), dst_id, indices)  # cell->gene
        indices = np.where((dst_id >= 0) & (src_id >= 0), self.gene_num, indices)  # gene-gene
        h = edges.src['h'] * self.alpha[indices.squeeze()]
        return {'m': h}

    #         return {'m': h * edges.data['weight']}

    def forward(self, graph, feat, weight=None, alpha=None, gene_num=None):
        self.alpha = alpha
        self.gene_num = gene_num
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            #             print(f"feat_src : {feat_src.shape}, feat_dst {feat_dst.shape}")
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1, ) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
                else:
                    feat_src = torch.matmul(feat_src, weight)
            else:
                weight = self.weight

            graph.srcdata['h'] = feat_src
            graph.update_all(self.edge_selection_simple, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            if self._norm != 'none':

                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
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
    data :
        data for leiden

    Returns
    -------
    pred : list
        prediction of leiden.

    """
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=300, n_pcs=0)
    sc.tl.leiden(adata)
    pred = adata.obs['leiden'].to_list()
    pred = [int(x) for x in pred]
    return pred


def get_device(use_cpu=False):
    """Get device for training.

    Parameters
    ----------
    use_cpu : bool optional
        use cpu or not.

    Returns
    -------
    device :
        torch device.

    """
    if torch.cuda.is_available() and use_cpu == False:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
