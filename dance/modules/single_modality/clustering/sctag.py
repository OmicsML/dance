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
from sklearn.cluster import KMeans
from torch.nn import Parameter

from dance import logger
from dance.modules.base import BaseClusteringMethod, TorchNNPretrain
from dance.transforms import AnnDataTransform, CellPCA, Compose, SaveRaw, SetConfig
from dance.transforms.graph import NeighborGraph
from dance.typing import Any, LogLevel, Optional, Tuple
from dance.utils import get_device
from dance.utils.loss import ZINBLoss, dist_loss


class ScTAG(nn.Module, TorchNNPretrain, BaseClusteringMethod):
    """The scTAG clustering model.

    Parameters
    ----------
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
    pretrain_path
        Path to save the pretrained autoencoder. If not specified, then do not save/load.

    """

    def __init__(
        self,
        n_clusters: int,
        k: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 15,
        dec_dim: Optional[int] = None,
        dropout: float = 0.2,
        device: str = "cuda",
        alpha: float = 1.0,
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        self._is_pretrained = False
        self._in_dim = None
        self.pretrain_path = pretrain_path

        self.dec_dim = dec_dim or [128, 256, 512]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = get_device(device)
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.k = k

    def init_model(self, adj: np.ndarray, x: np.ndarray):
        """Initialize model."""
        self._in_dim = x.shape[1]

        src, dist = np.nonzero(adj)
        # TODO: make util function for normalizing adj
        deg = adj.sum(1, keepdims=True)
        deg[deg == 0] = 1
        normalized_deg = deg**-0.5
        adj_n = adj * normalized_deg * normalized_deg.T
        src_n, dist_n = np.nonzero(adj_n)

        self.g = dgl.graph((src, dist)).to(self.device)
        self.g_n = dgl.graph((src_n, dist_n)).to(self.device)
        self.g_n.edata["weight"] = torch.FloatTensor(adj_n[src_n, dist_n]).to(self.device)

        self.mu = Parameter(torch.Tensor(self.n_clusters, self.latent_dim).to(self.device))
        self.encoder1 = TAGConv(self.in_dim, self.hidden_dim, k=self.k)
        self.encoder2 = TAGConv(self.hidden_dim, self.latent_dim, k=self.k)
        self.decoder_adj = DecoderAdj(latent_dim=self.latent_dim, adj_dim=adj.shape[0], activation=torch.sigmoid,
                                      dropout=self.dropout)
        self.decoder_x = DecoderX(self.in_dim, self.latent_dim, n_dec_1=self.dec_dim[0], n_dec_2=self.dec_dim[1],
                                  n_dec_3=self.dec_dim[2])
        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(self.device)

    @property
    def in_dim(self) -> int:
        if self._in_dim is None:
            raise ValueError("in_dim is unavailable since the model has not been initialized yet. Please call the "
                             "`fit` function first to fit the model, or the `init_model` function "
                             "if you just want to initialize the model.")
        return self._in_dim

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
                "feature_channel": ["NeighborGraph", None, None, "n_counts"],
                "feature_channel_type": ["obsp", "X", "raw_X", "obs"],
                "label_channel": "Group",
            }),
            log_level=log_level,
        )

    def forward(self, g, x_input):
        """Forward propagation.

        Parameters
        ----------
        g
            Input graph.
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
        enc_h = self.encoder1(g, x_input, edge_weight=g.edata["weight"])
        z = self.encoder2(g, enc_h, edge_weight=g.edata["weight"])
        adj_out = self.decoder_adj(z)
        _mean, _disp, _pi = self.decoder_x(z)
        q = self.soft_assign(z)

        return adj_out, z, q, _mean, _disp, _pi

    def pretrain(
        self,
        adj,
        x,
        x_raw,
        n_counts,
        *,
        epochs: int = 1000,
        info_step: int = 10,
        lr: float = 5e-4,
        w_a: float = 0.3,
        w_x: float = 1,
        w_d: float = 0,
        min_dist: float = 0.5,
        max_dist: float = 20,
        force_pretrain: bool = False,
    ):
        """Pretrain autoencoder.

        Parameters
        ----------
        adj
            Adjacency matrix.
        x
            Input features.
        x_raw
            Raw input features.
        n_counts
            Total counts for each cell.
        epochs
            Number of epochs.
        info_step
            Interval of showing pretraining loss.
        lr
            Learning rate.
        w_a
            Parameter of reconstruction loss.
        w_x
            Parameter of ZINB loss.
        w_d
            Parameter of pairwise distance loss.
        min_dist
            Minimum distance of pairwise distance loss.
        max_dist
            Maximum distance of pairwise distance loss.
        force_pretrain
            If set to True, then pre-train the model even if the pre-training has been done already,
            or even the pre-trained model file is available to load.

        """
        x = torch.Tensor(x).to(self.device)
        x_raw = torch.Tensor(x_raw).to(self.device)
        scale_factor = torch.tensor(n_counts / np.median(n_counts)).to(self.device)

        self.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        logger.info("Pre-training start")
        for epoch in range(epochs):
            adj_out, z, _, mean, disp, pi = self.forward(self.g_n, x)

            if w_d:
                dl = torch.mean(dist_loss(z, min_dist, max_dist=max_dist))
            adj_rec_loss = torch.mean(F.mse_loss(adj_out, torch.Tensor(adj).to(self.device)))
            Zinb_loss = self.zinb_loss(x_raw, mean, disp, pi, scale_factor)
            loss = w_a * adj_rec_loss + w_x * Zinb_loss
            if w_d:
                loss += w_d * dl

            if epoch % info_step == 0:
                if w_d:
                    logger.info("Epoch %3d: ZINB Loss: %.8f, MSE Loss: %.8f, Dist Loss: %.8f", epoch + 1,
                                Zinb_loss.item(), adj_rec_loss.item(), dl.item())
                else:
                    logger.info("Epoch %3d: ZINB Loss: %.8f, MSE Loss: %.8f", epoch + 1, Zinb_loss.item(),
                                adj_rec_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fit(
        self,
        inputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        y: np.ndarray,
        *,
        epochs: int = 300,
        pretrain_epochs: int = 200,
        lr: float = 5e-4,
        w_a: float = 0.3,
        w_x: float = 1,
        w_c: float = 1.5,
        w_d: float = 0,
        info_step: int = 1,
        max_dist: float = 20,
        min_dist: float = 0.5,
        force_pretrain: bool = False,
    ):
        """Pretrain autoencoder.

        Parameters
        ----------
        inputs
            A tuple containing the adjacency matrix, the input feature, the raw input feature, and the total counts per
            cell array.
        epochs
            Number of epochs.
        lr
            Learning rate.
        w_a
            Parameter of reconstruction loss.
        w_x
            Parameter of ZINB loss.
        w_c
            Parameter of clustering loss.
        w_d
            Parameter of pairwise distance loss.
        info_step
            Interval of showing pretraining loss.
        min_dist
            Minimum distance of pairwise distance loss.
        max_dist
            Maximum distance of pairwise distance loss.
        force_pretrain
            If set to True, then pre-train the model even if the pre-training has been done already,
            or even the pre-trained model file is available to load.

        """
        adj, x, x_raw, n_counts = inputs
        self.init_model(adj, x)
        self._pretrain(adj, x, x_raw, n_counts, epochs=pretrain_epochs, info_step=info_step, lr=lr, w_a=w_a, w_x=w_x,
                       w_d=w_d, min_dist=min_dist, max_dist=max_dist, force_pretrain=True)

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
            ari = self.score(None, y)
            aris.append(ari)
            p_ = {f"epoch{epoch}": p}
            q_ = {f"epoch{epoch}": q}
            P = {**P, **p_}
            Q = {**Q, **q_}
            if epoch % info_step == 0:
                logger.info("Epoch %3d, ARI: %.4f, Best ARI: %.4f", epoch + 1, ari, max(aris))

            adj_rec_loss = torch.mean(F.mse_loss(adj_out, torch.Tensor(adj).to(self.device)))
            Zinb_loss = self.zinb_loss(x_raw, mean, disp, pi, scale_factor)
            Cluster_loss = torch.mean(
                F.kl_div(
                    torch.Tensor(self.y_pred).to(self.device),
                    torch.Tensor(y).to(self.device), reduction="batchmean"))
            loss = w_a * adj_rec_loss + w_x * Zinb_loss + w_c * Cluster_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        index = np.argmax(aris)
        self.q = Q[f"epoch{index}"]

    def predict_proba(self, x: Optional[Any] = None) -> np.ndarray:
        """Get predicted probabilities for each cell.

        Parameters
        ----------
        x
            Not used, for compatibility with the base module class.

        Returns
        -------
        pred_prob
            Predicted probabilities for each cell.

        """
        pred_prob = self.q.detach().clone().cpu().numpy()
        return pred_prob

    def predict(self, x: Optional[Any] = None) -> np.ndarray:
        """Get predictions from the trained model.

        Parameters
        ----------
        x
            Not used, for compatibility with the base module class.

        Returns
        -------
        pred
            Prediction of given clustering method.

        """
        pred = self.predict_proba().argmax(1)
        return pred

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
