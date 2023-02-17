"""Reimplementation of scTAG.

Extended from https://github.com/Philyzh8/scTAG

Reference
----------
Yu, Z., Y. Lu, Y. Wang, F. Tang, K.-C. Wong, and X. Li. “ZINB-Based Graph Embedding Autoencoder for Single-Cell RNA-Seq
Interpretations”. Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 4, June 2022, pp. 4671-9,
doi:10.1609/aaai.v36i4.20392.

"""

import dgl
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import TAGConv
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.nn import Parameter

from dance import logger
from dance.transforms import AnnDataTransform, CellPCA, Compose, SaveRaw, SetConfig
from dance.transforms.graph import NeighborGraph
from dance.typing import LogLevel
from dance.utils.loss import ZINBLoss, dist_loss
from dance.utils.metrics import cluster_acc


class ScTAG(nn.Module):
    """The scTAG clustering model.

    Parameters
    ----------
    x
        Input features.
    adj
        Adjacency matrix.
    n_clusters
        Number of clusters.
    k
        Number of hops of TAG convolutional layer.
    hidden_dim
        Dimension of hidden layer.
    latent_dim
        Dimension of latent embedding.
    dec_dim
        Dimensions of decoder layers.
    dropout
        Dropout rate.
    device
        Computing device.
    alpha
        Parameter of soft assign.

    """

    def __init__(self, x, adj, n_clusters, k=3, hidden_dim=128, latent_dim=15, dec_dim=None, dropout=0.2, device="cuda",
                 alpha=1.0):
        super().__init__()
        if dec_dim is None:
            dec_dim = [128, 256, 512]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.adj = adj
        self.n_sample = x.shape[0]
        self.in_dim = x.shape[1]
        self.device = device
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.k = k
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.latent_dim).to(self.device))

        src, dist = np.nonzero(adj)
        self.g = dgl.graph((src, dist)).to(device)

        deg = adj.sum(1, keepdims=True)
        deg[deg == 0] = 1
        normalized_deg = deg**-0.5
        adj_n = adj * normalized_deg * normalized_deg.T
        src_n, dist_n = np.nonzero(adj_n)

        self.g_n = dgl.graph((src_n, dist_n)).to(device)

        self.encoder1 = TAGConv(self.in_dim, self.hidden_dim, k=k)
        self.encoder2 = TAGConv(self.hidden_dim, self.latent_dim, k=k)
        self.decoder_adj = DecoderAdj(latent_dim=self.latent_dim, adj_dim=adj.shape[0], activation=torch.sigmoid,
                                      dropout=self.dropout)
        self.decoder_x = DecoderX(self.in_dim, self.latent_dim, n_dec_1=dec_dim[0], n_dec_2=dec_dim[1],
                                  n_dec_3=dec_dim[2])
        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(self.device)

    @staticmethod
    def preprocessing_pipeline(n_top_genes: int = 3000, n_components: int = 50, n_neighbors: int = 15,
                               log_level: LogLevel = "INFO"):
        return Compose(
            # Filter data
            AnnDataTransform(sc.pp.filter_genes, min_counts=3),
            AnnDataTransform(sc.pp.filter_cells, min_counts=1),
            AnnDataTransform(sc.pp.normalize_per_cell),
            AnnDataTransform(sc.pp.log1p),
            AnnDataTransform(sc.pp.highly_variable_genes, min_mean=0.0125, max_mean=4, flavor="cell_ranger",
                             min_disp=0.5, n_top_genes=n_top_genes, subset=True),
            # Normalize data
            AnnDataTransform(sc.pp.filter_genes, min_counts=1),
            AnnDataTransform(sc.pp.filter_cells, min_counts=1),
            SaveRaw(),
            AnnDataTransform(sc.pp.normalize_total),
            AnnDataTransform(sc.pp.log1p),
            AnnDataTransform(sc.pp.scale),
            # Construct k-neighbors graph
            CellPCA(n_components=n_components),
            NeighborGraph(n_neighbors=n_neighbors, n_pcs=n_components),
            SetConfig({
                "feature_channel": [None, None, "n_counts", "NeighborGraph"],
                "feature_channel_type": ["X", "raw_X", "obs", "obsp"],
                "label_channel": "Group",
            }),
            log_level=log_level,
        )

    def forward(self, adj_in, x_input):
        """Forward propagation.

        Parameters
        ----------
        adj_in
            Input adjacency matrix.
        x_input
            Input features.

        Returns
        -------
        adj_out
            Reconstructed adjacency matrix.
        z
            Embedding.
        q
            Soft label.
        _mean
            Data mean from ZINB.
        _disp
            Data dispersion from ZINB.
        _pi
            Data dropout probability from ZINB.

        """
        enc_h = self.encoder1(adj_in, x_input)
        z = self.encoder2(adj_in, enc_h)
        adj_out = self.decoder_adj(z)
        _mean, _disp, _pi = self.decoder_x(z)
        q = self.soft_assign(z)

        return adj_out, z, q, _mean, _disp, _pi

    def pre_train(self, x, x_raw, fname, n_counts, epochs=1000, info_step=10, lr=5e-4, W_a=0.3, W_x=1, W_d=0,
                  min_dist=0.5, max_dist=20.):
        """Pretrain autoencoder.

        Parameters
        ----------
        x
            Input features.
        x_raw
            Raw input features.
        fname
            Path to save autoencoder weights.
        n_counts
            Total counts for each cell.
        epochs
            Number of epochs.
        info_step
            Interval of showing pretraining loss.
        lr
            Learning rate.
        W_a
            Parameter of reconstruction loss.
        W_x
            Parameter of ZINB loss.
        W_d
            Parameter of pairwise distance loss.
        min_dist
            Minimum distance of pairwise distance loss.
        max_dist
            Maximum distance of pairwise distance loss.

        """
        x = torch.Tensor(x).to(self.device)
        x_raw = torch.Tensor(x_raw).to(self.device)
        scale_factor = torch.tensor(n_counts / np.median(n_counts)).to(self.device)

        self.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        logger.info("Pre-training start")
        for epoch in range(epochs):
            adj_out, z, _, mean, disp, pi = self.forward(self.g_n, x)

            if W_d:
                Dist_loss = torch.mean(dist_loss(z, min_dist, max_dist=max_dist))
            adj_rec_loss = torch.mean(F.mse_loss(adj_out, torch.Tensor(self.adj).to(self.device)))
            Zinb_loss = self.zinb_loss(x_raw, mean, disp, pi, scale_factor)
            loss = W_a * adj_rec_loss + W_x * Zinb_loss
            if W_d:
                loss += W_d * Dist_loss

            if epoch % info_step == 0:
                if W_d:
                    logger.info("Epoch %3d: ZINB Loss: %.8f, MSE Loss: %.8f, Dist Loss: %.8f", epoch + 1,
                                Zinb_loss.item(), adj_rec_loss.item(), Dist_loss.item())
                else:
                    logger.info("Epoch %3d: ZINB Loss: %.8f, MSE Loss: %.8f", epoch + 1, Zinb_loss.item(),
                                adj_rec_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(self.state_dict(), fname)
        logger.info("Pre-training done")
        self._is_pretrained = True

    def fit(
        self,
        x,
        x_raw,
        y,
        n_counts,
        *,
        epochs: int = 300,
        pretrain_epochs: int = 200,
        lr: float = 5e-4,
        W_a: float = 0.3,
        W_x: float = 1,
        W_c: float = 1.5,
        W_d: float = 0,
        info_step: int = 1,
        max_dist: float = 20,
        min_dist: float = 0.5,
    ):
        # FIX:  update docstirng
        # FIX: add pretrain save/load file option, with path
        """Pretrain autoencoder.

        Parameters
        ----------
        x
            Input features.
        x_raw
            Raw input features.
        y
            True label.
        n_counts
            Total counts for each cell.
        epochs
            Number of epochs.
        lr
            Learning rate.
        W_a
            Parameter of reconstruction loss.
        W_x
            Parameter of ZINB loss.
        W_c
            Parameter of clustering loss.
        info_step
            Interval of showing pretraining loss.

        """
        x = torch.Tensor(x).to(self.device)
        x_raw = torch.Tensor(x_raw).to(self.device)
        scale_factor = torch.tensor(n_counts / np.median(n_counts)).to(self.device)

        # Initializing cluster centers with kmeans
        kmeans = KMeans(self.n_clusters, n_init=20)
        enc_h = self.encoder1(self.g_n, x)
        z = self.encoder2(self.g_n, enc_h)
        kmeans.fit_predict(z.detach().cpu().numpy())
        self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device))

        self.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)

        aris = []
        P = {}
        Q = {}

        for epoch in range(epochs):
            adj_out, _, q, mean, disp, pi = self.forward(self.g_n, x)
            self.q = q
            p = self.target_distribution(q)
            self.y_pred = self.predict()

            # calculate ari score for model selection
            _, _, ari = self.score(y)
            aris.append(ari)
            p_ = {f"epoch{epoch}": p}
            q_ = {f"epoch{epoch}": q}
            P = {**P, **p_}
            Q = {**Q, **q_}
            if epoch % info_step == 0:
                logger.info("Epoch %3d, ARI: %.4f, Best ARI: %.4f", epoch + 1, ari, max(aris))

            adj_rec_loss = torch.mean(F.mse_loss(adj_out, torch.Tensor(self.adj).to(self.device)))
            Zinb_loss = self.zinb_loss(x_raw, mean, disp, pi, scale_factor)
            Cluster_loss = torch.mean(
                F.kl_div(
                    torch.Tensor(self.y_pred).to(self.device),
                    torch.Tensor(y).to(self.device), reduction="batchmean"))
            loss = W_a * adj_rec_loss + W_x * Zinb_loss + W_c * Cluster_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        index = np.argmax(aris)
        self.q = Q[f"epoch{index}"]

    def predict(self):
        """Get predictions from the trained model.

        Returns
        -------
        y_pred
            Prediction of given clustering method.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        return y_pred

    def score(self, y):
        """Evaluate the trained model.

        Parameters
        ----------
        y
            True labels.

        Returns
        -------
        acc
            Accuracy.
        nmi
            Normalized mutual information.
        ari
            Adjusted Rand index.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        return acc, nmi, ari

    def soft_assign(self, z):
        """Soft assign q with z.

        Parameters
        ----------
        z
            Embedding.

        Returns
        -------
        q
            Soft label.

        """
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        """Calculate auxiliary target distribution p with q.

        Parameters
        ----------
        q
            Soft label.

        Returns
        -------
        p
            Target distribution.

        """
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()


class DecoderAdj(nn.Module):
    """Decoder for adjacency matrix.

    Parameters
    ----------
    latent_dim
        Dimension of latent embedding.
    adj_dim
        Dimension of adjacency matrix.
    activation
        Activation function.
    dropout
        Dropout rate.

    """

    def __init__(self, latent_dim=15, adj_dim=32, activation=torch.sigmoid, dropout=0):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.dec_1 = nn.Linear(latent_dim, adj_dim)

    def forward(self, z):
        """Forward propagation.

        Parameters
        ----------
        z
            Embedding.

        Returns
        -------
        adj
            Reconstructed adjacency matrix.

        """
        dec_h = self.dec_1(z)
        z0 = F.dropout(dec_h, self.dropout)
        adj = self.activation(torch.mm(z0, z0.t()))
        return adj


class DecoderX(nn.Module):
    """Decoder for feature.

    Parameters
    ----------
    input_dim
        Dimension of input feature.
    n_z
        Dimension of latent embedding.
    n_dec_1
        Number of nodes of decoder layer 1.
    n_dec_2
        Number of nodes of decoder layer 2.
    n_dec_3
        Number of nodes of decoder layer 3.

    """

    def __init__(self, input_dim, n_z, n_dec_1=128, n_dec_2=256, n_dec_3=512):
        super().__init__()
        self.n_dec_3 = n_dec_3
        self.input_dim = input_dim
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.dec_mean = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), nn.Sigmoid())

    def forward(self, z):
        """Forward propagation.

        Parameters
        ----------
        z
            Embedding.

        Returns
        -------
        _mean
            Eata mean from ZINB.
        _disp
            Eata dispersion from ZINB.
        _pi
            Eata dropout probability from ZINB.

        """
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        _mean = self.dec_mean(dec_h3)
        _disp = self.dec_disp(dec_h3)
        _pi = self.dec_pi(dec_h3)
        return _mean, _disp, _pi


class MeanAct(nn.Module):
    """Mean activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Dispersion activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
