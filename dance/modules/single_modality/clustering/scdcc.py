"""Reimplementation of scDCC.

Extended from https://github.com/ttgump/scDCC

Reference
----------
Tian, Tian, et al. "Model-based deep embedding for constrained clustering analysis of single cell RNA-seq data."
Nature communications 12.1 (2021): 1-12.

"""
import math

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from dance import logger
from dance.modules.base import BaseClusteringMethod, TorchNNPretrain
from dance.transforms import AnnDataTransform, Compose, SaveRaw, SetConfig
from dance.typing import Any, List, LogLevel, Optional, Tuple
from dance.utils import get_device
from dance.utils.loss import ZINBLoss


def buildNetwork(layers: List[int], network_type: str, activation: str = "relu"):
    """Build network layer.

    Parameters
    ----------
    layers
        Dimensions of layers.
    network_type
        Type of network.
    activation
        Activation function.


    Returns
    -------
    Built network.

    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    net = nn.Sequential(*net)
    return net


class ScDCC(nn.Module, TorchNNPretrain, BaseClusteringMethod):
    """ScDCC class.

    Parameters
    ----------
    input_dim
        Dimension of encoder input.
    z_dim
        Dimension of embedding.
    n_clusters
        Number of clusters.
    encodeLayer
        Dimensions of encoder layers.
    decodeLayer
        Dimensions of decoder layers.
    activation
        Activation function.
    sigma
        Parameter of Gaussian noise.
    alpha
        Parameter of soft assign.
    gamma
        Parameter of cluster loss.
    ml_weight
        Parameter of must-link loss.
    cl_weight
        Parameter of cannot-link loss.
    device
        Computation device.

    """

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        n_clusters: int,
        encodeLayer: List[int],
        decodeLayer: List[int],
        activation: str = "relu",
        sigma: float = 1.,
        alpha: float = 1.,
        gamma: float = 1.,
        ml_weight: float = 1.,
        cl_weight: float = 1.,
        device: str = "auto",
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.device = get_device(device)
        self.pretrain_path = pretrain_path

        self.encoder = buildNetwork([input_dim] + encodeLayer, network_type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer, network_type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().to(self.device)

        self.to(self.device)

    @staticmethod
    def preprocessing_pipeline(log_level: LogLevel = "INFO"):
        return Compose(
            AnnDataTransform(sc.pp.filter_genes, min_counts=1),
            AnnDataTransform(sc.pp.filter_cells, min_counts=1),
            SaveRaw(),
            AnnDataTransform(sc.pp.normalize_total),
            AnnDataTransform(sc.pp.log1p),
            AnnDataTransform(sc.pp.scale),
            SetConfig({
                "feature_channel": [None, None, "n_counts"],
                "feature_channel_type": ["X", "raw_X", "obs"],
                "label_channel": "Group"
            }),
            log_level=log_level,
        )

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

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x
            Input features.

        Returns
        -------
        z0
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
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi

    def encodeBatch(self, X, batch_size=256):
        """Batch encoder.

        Parameters
        ----------
        X
            Input features.
        batch_size
            Size of batch.

        Returns
        -------
        Embedding.

        """
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
            inputs = xbatch
            z, _, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        """Calculate cluster loss.

        Parameters
        ----------
        p
            Target distribution.
        q
            Soft label.

        Returns
        -------
        Cluster loss.

        """

        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)
        loss = self.gamma * kldloss
        return loss

    def pairwise_loss(self, p1, p2, cons_type):
        """Calculate pairwise loss.

        Parameters
        ----------
        p1
            Distribution 1.
        p2
            Distribution 2.
        cons_type
            Type of loss.

        Returns
        -------
        Pairwise loss.

        """
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            loss = self.ml_weight * ml_loss
            return loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            loss = self.cl_weight * cl_loss
            return loss

    def pretrain(self, x, X_raw, n_counts, batch_size=256, lr=0.001, epochs=400):
        """Pretrain autoencoder.

        Parameters
        ----------
        x
            Input features.
        X_raw
            Raw input features.
        n_counts
            Total counts for each cell.
        batch_size
            Size of batch.
        lr
            Learning rate.
        epochs
            Number of epochs.

        """
        size_factor = torch.tensor(n_counts / np.median(n_counts))
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = x_batch.to(self.device)
                x_raw_tensor = x_raw_batch.to(self.device)
                sf_tensor = sf_batch.to(self.device)
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                      scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 100 == 0:
                # logger.info("Pretrain epoch [%2d/%3d], ZINB loss: %.4f", batch_idx + 1, epoch + 1, loss.item())
                logger.info("Pretrain epoch [%3d], ZINB loss: %.4f", epoch + 1, loss.item())

    def fit(
        self,
        inputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        y: np.ndarray = None,
        ml_ind1: np.ndarray = np.array([]),
        ml_ind2: np.ndarray = np.array([]),
        cl_ind1: np.ndarray = np.array([]),
        cl_ind2: np.ndarray = np.array([]),
        ml_p: float = 1.,
        cl_p: float = 1.,
        lr: float = 1.,
        batch_size: int = 256,
        epochs: int = 10,
        update_interval: int = 1,
        tol: float = 1e-3,
        pt_batch_size: int = 256,
        pt_lr: float = 0.001,
        pt_epochs: int = 400,
    ):
        """Train model.

        Parameters
        ----------
        inputs
            A tuple containing (1) the input features, (2) the raw input features, and (3) the total counts per cell.
        y
            True label. Used for model selection.
        ml_ind1
            Index 1 of must-link pairs.
        ml_ind2
            Index 2 of must-link pairs.
        cl_ind1
            Index 1 of cannot-link pairs.
        cl_ind2
            Index 2 of cannot-link pairs.
        ml_p
            Parameter of must-link loss.
        cl_p
            Parameter of cannot-link loss.
        lr
            Learning rate.
        batch_size
            Size of batch.
        epochs
            Number of epochs.
        update_interval
            Update interval of soft label and target distribution.
        tol
            Tolerance for training loss.
        pt_batch_size
            Pretrain batch size.
        pt_lr
            Pretrain learning rate.
        pt_epochs
            Pretrain epochs.

        """
        X, X_raw, n_counts = inputs
        self._pretrain(X, X_raw, n_counts, batch_size=pt_batch_size, lr=pt_lr, epochs=pt_epochs, force_pretrain=True)

        X = torch.tensor(X).to(self.device)
        X_raw = torch.tensor(X_raw).to(self.device)
        sf = torch.tensor(n_counts / np.median(n_counts)).to(self.device)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        # Initializing cluster centers with kmeans
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        ml_num_batch = int(math.ceil(1.0 * ml_ind1.shape[0] / batch_size))
        cl_num_batch = int(math.ceil(1.0 * cl_ind1.shape[0] / batch_size))
        cl_num = cl_ind1.shape[0]
        ml_num = ml_ind1.shape[0]

        update_ml = 1
        update_cl = 1

        aris = []
        P = {}
        Q = {}

        delta_label = np.inf
        for epoch in range(epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                self.q = q
                p = self.target_distribution(q).data
                self.y_pred = self.predict()

                p_ = {f"epoch{epoch}": p}
                q_ = {f"epoch{epoch}": q}
                P = {**P, **p_}
                Q = {**Q, **q_}

                # check stop criterion
                if False:
                    delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                    self.y_pred_last = self.y_pred
                    if epoch > 0 and delta_label < tol:
                        logger.info("Reach tolerance threshold (%.3e < %.3e). Stopping training.", delta_label, tol)
                        break

                # calculate ari score for model selection
                ari = self.score(None, y)
                aris.append(ari)

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                xrawbatch = X_raw[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                sfbatch = sf[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = xbatch
                rawinputs = xrawbatch
                sfinputs = sfbatch
                target = pbatch

                z, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss = cluster_loss + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val

            if epoch % 50 == 0:
                logger.info("#Epoch %3d: Total: %.4f, Clustering Loss: %.4f, ZINB Loss: %.4f", epoch + 1,
                            train_loss / num, cluster_loss_val / num, recon_loss_val / num)

            ml_loss = 0.0
            if epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[ml_ind1[ml_batch_idx * batch_size:min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw1 = X_raw[ml_ind1[ml_batch_idx * batch_size:min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf1 = sf[ml_ind1[ml_batch_idx * batch_size:min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    px2 = X[ml_ind2[ml_batch_idx * batch_size:min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf2 = sf[ml_ind2[ml_batch_idx * batch_size:min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw2 = X_raw[ml_ind2[ml_batch_idx * batch_size:min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = px1
                    rawinputs1 = pxraw1
                    sfinput1 = sf1
                    inputs2 = px2
                    rawinputs2 = pxraw2
                    sfinput2 = sf2
                    z1, q1, mean1, disp1, pi1 = self.forward(inputs1)
                    z2, q2, mean2, disp2, pi2 = self.forward(inputs2)
                    loss = (ml_p * self.pairwise_loss(q1, q2, "ML") +
                            self.zinb_loss(rawinputs1, mean1, disp1, pi1, sfinput1) +
                            self.zinb_loss(rawinputs2, mean2, disp2, pi2, sfinput2))
                    # 0.1 for mnist/reuters, 1 for fashion, the parameters are tuned via grid search on validation set
                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()

            cl_loss = 0.0
            if epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    px1 = X[cl_ind1[cl_batch_idx * batch_size:min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    px2 = X[cl_ind2[cl_batch_idx * batch_size:min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = px1
                    inputs2 = px2
                    z1, q1, _, _, _ = self.forward(inputs1)
                    z2, q2, _, _, _ = self.forward(inputs2)
                    loss = cl_p * self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch > 0 and cl_num_batch > 0:
                if epoch % 50 == 0:
                    logger.info("Pairwise Total: %.4f, ML loss: %.4f, CL loss: %.4f",
                                float(ml_loss.cpu()) + float(cl_loss.cpu()), ml_loss.cpu(), cl_loss.cpu())

        index = update_interval * np.argmax(aris)
        self.q = Q[f"epoch{index}"]

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
