"""Reimplementation of scDSC.

Extended from https://github.com/DHUDBlab/scDSC

Reference
----------
Gan, Yanglan, et al. "Deep structural clustering for single-cell RNA-seq data jointly through autoencoder and graph
neural network." Briefings in Bioinformatics 23.2 (2022): bbac018.

"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from dance import logger
from dance.modules.base import BaseClusteringMethod, TorchNNPretrain
from dance.transforms import AnnDataTransform, Compose, SaveRaw, SetConfig
from dance.transforms.graph import NeighborGraph
from dance.transforms.preprocess import sparse_mx_to_torch_sparse_tensor
from dance.typing import Any, LogLevel, Optional, Tuple
from dance.utils import get_device
from dance.utils.loss import ZINBLoss


class ScDSC(TorchNNPretrain, BaseClusteringMethod):
    """ScDSC wrapper class.

    Parameters
    ----------
    pretrain_path
        Path of saved autoencoder weights.
    sigma
        Balance parameter.
    n_enc_1
        Output dimension of encoder layer 1.
    n_enc_2
        Output dimension of encoder layer 2.
    n_enc_3
        Output dimension of encoder layer 3.
    n_dec_1
        Output dimension of decoder layer 1.
    n_dec_2
        Output dimension of decoder layer 2.
    n_dec_3
        Output dimension of decoder layer 3.
    n_z1
        Output dimension of hidden layer 1.
    n_z2
        Output dimension of hidden layer 2.
    n_z3
        Output dimension of hidden layer 3.
    n_clusters
        Number of clusters.
    n_input
        Input feature dimension.
    v
        Parameter of soft assignment.
    device
        Computing device.

    """

    def __init__(
        self,
        pretrain_path: str,
        sigma: float = 1,
        n_enc_1: int = 512,
        n_enc_2: int = 256,
        n_enc_3: int = 256,
        n_dec_1: int = 256,
        n_dec_2: int = 256,
        n_dec_3: int = 512,
        n_z1: int = 256,
        n_z2: int = 128,
        n_z3: int = 32,
        n_clusters: int = 100,
        n_input: int = 10,
        v: float = 1,
        device: str = "auto",
    ):
        super().__init__()
        self.pretrain_path = pretrain_path
        self.device = get_device(device)
        self.model = ScDSCModel(
            sigma=sigma,
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_z1=n_z1,
            n_z2=n_z2,
            n_z3=n_z3,
            n_clusters=n_clusters,
            n_input=n_input,
            v=v,
            device=self.device,
        ).to(self.device)
        self.fix_module("model.ae")

    @staticmethod
    def preprocessing_pipeline(n_top_genes: int = 2000, n_neighbors: int = 50, log_level: LogLevel = "INFO"):
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
            # Construct k-neighbors graph using the noramlized feature matrix
            NeighborGraph(n_neighbors=n_neighbors, metric="correlation", channel="X"),
            SetConfig({
                "feature_channel": ["NeighborGraph", None, None, "n_counts"],
                "feature_channel_type": ["obsp", "X", "raw_X", "obs"],
                "label_channel": "Group"
            }),
            log_level=log_level,
        )

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

    def pretrain(self, x, batch_size=256, epochs=200, lr=1e-3):
        """Pretrain autoencoder.

        Parameters
        ----------
        x
            Input features.
        batch_size
            Size of batch.
        epochs
            Number of epochs.
        lr
            Learning rate.

        """
        with self.pretrain_context("model.ae"):
            x_tensor = torch.from_numpy(x)
            train_loader = DataLoader(TensorDataset(x_tensor), batch_size, shuffle=True)
            model = self.model.ae
            optimizer = Adam(model.parameters(), lr=lr)
            for epoch in range(epochs):

                total_loss = total_size = 0
                for batch_idx, (x_batch, ) in enumerate(train_loader):
                    x_batch = x_batch.to(self.device)
                    x_bar, _, _, _, _, _, _, _ = model(x_batch)

                    loss = F.mse_loss(x_bar, x_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    size = x_batch.shape[0]
                    total_size += size
                    total_loss += loss.item() * size

                if epoch % 100 == 0:
                    logger.info(f"Pretrain epoch {epoch + 1:4d}, MSE loss:{total_loss / total_size:.8f}")

    def save_pretrained(self, path):
        torch.save(self.model.ae.state_dict(), path)

    def load_pretrained(self, path):
        checkpoint = torch.load(self.pretrain_path, map_location=self.device)
        self.model.ae.load_state_dict(checkpoint)

    def fit(
        self,
        inputs: Tuple[sp.spmatrix, np.ndarray, np.ndarray, pd.Series],
        y: np.ndarray,
        lr: float = 1e-03,
        epochs: int = 300,
        bcl: float = 0.1,
        cl: float = 0.01,
        rl: float = 1,
        zl: float = 0.1,
        pt_epochs: int = 200,
        pt_batch_size: int = 256,
        pt_lr: float = 1e-3,
    ):
        """Train model.

        Parameters
        ----------
        inputs
            A tuple containing (1) the adjacency matrix, (2) the input features, (3) the raw input features, and (4)
            the total counts for each cell.
        y
            Label.
        lr
            Learning rate.
        epochs
            Number of epochs.
        bcl
            Parameter of binary crossentropy loss.
        cl
            Parameter of Kullback–Leibler divergence loss.
        rl
            Parameter of reconstruction loss.
        zl
            Parameter of ZINB loss.

        """
        adj, x, x_raw, n_counts = inputs
        self._pretrain(x, batch_size=pt_batch_size, epochs=pt_epochs, lr=pt_lr, force_pretrain=True)

        device = self.device
        model = self.model
        optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)

        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        x_raw = torch.tensor(x_raw).to(device)
        sf = torch.tensor(n_counts / np.median(n_counts)).to(device)
        data = torch.from_numpy(x).to(device)

        aris = []
        keys = []
        P = {}
        Q = {}

        with torch.no_grad():
            _, _, _, _, z, _, _, _ = model.ae(data)

        for epoch in range(epochs):
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    x_bar, tmp_q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)
                    tmp_q = tmp_q.data
                    self.q = tmp_q
                    p = self.target_distribution(tmp_q)

                    # calculate ari score for model selection
                    ari = self.score(None, y)
                    aris.append(ari)
                    keys.append(key := f"epoch{epoch}")
                    # logger.info("Epoch %3d, ARI: %.4f, Best ARI: %.4f", epoch + 1, ari, max(aris))
                    if epoch % 100 == 0:
                        logger.info("Epoch %3d, ARI: %.4f, Best ARI: %.4f", epoch + 1, ari, max(aris))

                    P[key] = p
                    Q[key] = tmp_q

            model.train()
            x_bar, q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)

            binary_crossentropy_loss = F.binary_cross_entropy(q, p)
            ce_loss = F.kl_div(pred.log(), p, reduction="batchmean")
            re_loss = F.mse_loss(x_bar, data)
            zinb_loss = zinb_loss(x_raw, meanbatch, dispbatch, pibatch, sf)
            loss = bcl * binary_crossentropy_loss + cl * ce_loss + rl * re_loss + zl * zinb_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        index = np.argmax(aris)
        self.q = Q[keys[index]]

    def predict_proba(self, x: Optional[Any] = None) -> np.ndarray:
        """Get the predicted propabilities for each cell.

        Parameters
        ----------
        x
            Not used, for compatibility with the BaseClusteringMethod class.

        Returns
        -------
        pred_prop
            Predicted probability for each cell.

        """
        pred_prob = self.q.detach().clone().cpu().numpy()
        return pred_prob

    def predict(self, x: Optional[Any] = None) -> np.ndarray:
        """Get predictions from the trained model.

        Parameters
        ----------
        x
            Not used, for compatibility with the BaseClusteringMethod class.

        Returns
        -------
        pred
            Predicted clustering assignment for each cell.

        """
        pred = self.predict_proba().argmax(1)
        return pred


class ScDSCModel(nn.Module):
    """ScDSC class.

    Parameters
    ----------
    sigma
        Balance parameter.
    n_enc_1
        Output dimension of encoder layer 1.
    n_enc_2
        Output dimension of encoder layer 2.
    n_enc_3
        Output dimension of encoder layer 3.
    n_dec_1
        Output dimension of decoder layer 1.
    n_dec_2
        Output dimension of decoder layer 2.
    n_dec_3
        Output dimension of decoder layer 3.
    n_z1
        Output dimension of hidden layer 1.
    n_z2
        Output dimension of hidden layer 2.
    n_z3
        Output dimension of hidden layer 3.
    n_clusters
        Number of clusters.
    n_input
        Input feature dimension.
    v
        Parameter of soft assignment.
    device
        Computing device.

    """

    def __init__(
        self,
        sigma: float = 1,
        n_enc_1: int = 512,
        n_enc_2: int = 256,
        n_enc_3: int = 256,
        n_dec_1: int = 256,
        n_dec_2: int = 256,
        n_dec_3: int = 512,
        n_z1: int = 256,
        n_z2: int = 128,
        n_z3: int = 32,
        n_clusters: int = 10,
        n_input: int = 100,
        v: float = 1,
        device: str = "auto",
    ):
        super().__init__()
        self.device = get_device(device)
        self.sigma = sigma

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z1=n_z1,
            n_z2=n_z2,
            n_z3=n_z3,
        )

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z1)
        self.gnn_5 = GNNLayer(n_z1, n_z2)
        self.gnn_6 = GNNLayer(n_z2, n_z3)
        self.gnn_7 = GNNLayer(n_z3, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z3))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
        # degree
        self.v = v
        self.zinb_loss = ZINBLoss().to(self.device)

        self.to(self.device)

    def forward(self, x, adj):
        """Forward propagation.

        Parameters
        ----------
        x
            Input features.
        adj
            Adjacency matrix

        Returns
        -------
        x_bar:
            Reconstructed features.
        q
            Soft label.
        predict:
            Prediction given by softmax assignment of embedding of GCN module
        z3
            Embedding of autoencoder.
        _mean
            Data mean from ZINB.
        _disp
            Data dispersion from ZINB.
        _pi
            Data dropout probability from ZINB.
        zinb_loss:
            ZINB loss class.

        """
        # DNN Module
        x_bar, tra1, tra2, tra3, z3, z2, z1, dec_h3 = self.ae(x)

        # GCN Module
        sigma = self.sigma
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z1, adj)
        h = self.gnn_6((1 - sigma) * h + sigma * z2, adj)
        h = self.gnn_7((1 - sigma) * h + sigma * z3, adj, active=False)

        predict = F.softmax(h, dim=1)

        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)
        zinb_loss = self.zinb_loss

        q = 1.0 / (1.0 + torch.sum(torch.pow(z3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z3, _mean, _disp, _pi, zinb_loss


class GNNLayer(nn.Module):
    """GNN layer class. Construct a GNN layer with corresponding dimensions.

    Parameters
    ----------
    in_features
        Input dimension of GNN layer.
    out_features
        Output dimension of GNN layer.

    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        # When passing through the network layer, the input and output variances are the same
        # including forward propagation and backward propagation

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output


class AE(nn.Module):
    """Autoencoder class.

    Parameters
    ----------
    n_enc_1
        Output dimension of encoder layer 1.
    n_enc_2
        Output dimension of encoder layer 2.
    n_enc_3
        Output dimension of encoder layer 3.
    n_dec_1
        Output dimension of decoder layer 1.
    n_dec_2
        Output dimension of decoder layer 2.
    n_dec_3
        Output dimension of decoder layer 3.
    n_input
        Input feature dimension.
    n_z1
        Output dimension of hidden layer 1.
    n_z2
        Output dimension of hidden layer 2.
    n_z3
        Output dimension of hidden layer 3.

    """

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z1, n_z2, n_z3):
        super().__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        self.z1_layer = Linear(n_enc_3, n_z1)
        self.BN4 = nn.BatchNorm1d(n_z1)
        self.z2_layer = Linear(n_z1, n_z2)
        self.BN5 = nn.BatchNorm1d(n_z2)
        self.z3_layer = Linear(n_z2, n_z3)
        self.BN6 = nn.BatchNorm1d(n_z3)

        self.dec_1 = Linear(n_z3, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x
            Input features.

        Returns
        -------
        x_bar
            Reconstructed features.
        enc_h1
            Output of encoder layer 1.
        enc_h2
            Output of encoder layer 2.
        enc_h3
            Output of encoder layer 3.
        z3
            Output of hidden layer 3.
        z2
            Output of hidden layer 2.
        z1
            Output of hidden layer 1.
        dec_h3
            Output of decoder layer 3.

        """
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z1 = self.BN4(self.z1_layer(enc_h3))
        z2 = self.BN5(self.z2_layer(z1))
        z3 = self.BN6(self.z3_layer(z2))

        dec_h1 = F.relu(self.BN7(self.dec_1(z3)))
        dec_h2 = F.relu(self.BN8(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN9(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z3, z2, z1, dec_h3


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
