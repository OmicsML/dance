"""Reimplementation of scDeepCluster.

Extended from https://github.com/ttgump/scDeepCluster

Reference
----------
Tian, Tian, et al. "Clustering single-cell RNA-seq data with a model-based deep learning approach." Nature Machine
Intelligence 1.4 (2019): 191-198.

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


def euclidean_dist(x, y):
    """Calculate Euclidean distance between x and y."""
    return torch.sum(torch.square(x - y), dim=1)


class ScDeepCluster(nn.Module, TorchNNPretrain, BaseClusteringMethod):
    """ScDeepCluster class.

    Parameters
    ----------
    input_dim
        Dimension of encoder input.
    z_dim
        Dimension of embedding.
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
    device
        Computing device.
    pretrain_path
        Path to pretrained weights.

    """

    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], activation="relu", sigma=1., alpha=1.,
                 gamma=1., device="cuda", pretrain_path: Optional[str] = None):
        super().__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.pretrain_path = pretrain_path

        self.encoder = buildNetwork([input_dim] + encodeLayer, network_type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer, network_type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)

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
                "label_channel": "Group",
            }),
            log_level=log_level,
        )

    def save_model(self, path):
        """Save model to path.

        Parameters
        ----------
        path
            Path to save model.

        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model from path.

        Parameters
        ----------
        path
            Path to load model.

        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

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

    def forwardAE(self, x):
        """Forward propagation of autoencoder.

        Parameters
        ----------
        x
            Input features.

        Returns
        -------
        z0
            Embedding.
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
        return z0, _mean, _disp, _pi

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

    def encodeBatch(self, x, batch_size=256):
        """Batch encoder.

        Parameters
        ----------
        x
            Input features.
        batch_size
            Size of batch.

        Returns
        -------
        encoded
            Embedding.

        """
        self.eval()
        encoded = []
        num = x.shape[0]
        num_batch = int(math.ceil(1.0 * x.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = x[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
            inputs = xbatch.to(self.device)
            z, _, _, _ = self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)

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
        loss
            Cluster loss.

        """

        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)
        return self.gamma * kldloss

    def pretrain(self, x, x_raw, n_counts, batch_size=256, lr=0.001, epochs=400):
        """Pretrain autoencoder.

        Parameters
        ----------
        x
            Input features.
        x_raw
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
        self.train()
        size_factor = torch.tensor(n_counts / np.median(n_counts))
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(x_raw), size_factor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = x_batch.to(self.device)
                x_raw_tensor = x_raw_batch.to(self.device)
                sf_tensor = sf_batch.to(self.device)
                _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                      scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item() * len(x_batch)
            if epoch % 100 == 0:
                logger.info("Pretrain epoch %3d, ZINB loss: %.8f", epoch + 1, loss_val / x.shape[0])

    def fit(
        self,
        inputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        y: np.ndarray,
        n_clusters: int = 10,
        init_centroid: Optional[List[int]] = None,
        y_pred_init: Optional[List[int]] = None,
        lr: float = 1,
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
        n_clusters
            Number of clusters.
        init_centroid
            Initialization of centroids. If None, perform kmeans to initialize cluster centers.
        y_pred_init
            Predicted label for initialization.
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
            Pretraining batch size.
        pt_lr
            Pretraining learning rate.
        pt_epochs
            pretraining epochs.

        """
        x, x_raw, n_counts = inputs
        self._pretrain(x, x_raw, n_counts, batch_size=pt_batch_size, lr=pt_lr, epochs=pt_epochs, force_pretrain=True)

        self.train()
        x = torch.tensor(x, dtype=torch.float32)
        x_raw = torch.tensor(x_raw, dtype=torch.float32)
        size_factor = torch.FloatTensor(n_counts / np.median(n_counts))
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        logger.info("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(x)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float32))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred

        num = x.shape[0]
        num_batch = int(math.ceil(1.0 * x.shape[0] / batch_size))

        aris = []
        P = {}
        Q = {}

        delta_label = np.inf
        for epoch in range(epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(x.to(self.device))
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
                xbatch = x[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                xrawbatch = x_raw[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                sfbatch = size_factor[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = xbatch.to(self.device)
                rawinputs = xrawbatch.to(self.device)
                sfinputs = sfbatch.to(self.device)
                target = pbatch.to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                loss = cluster_loss * self.gamma + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                recon_loss_val += recon_loss.item() * len(inputs)
                train_loss += loss.item() * len(inputs)

            if epoch % 50 == 0:
                logger.info("Epoch %3d: Total: %.8f, Clustering Loss: %.8f, ZINB Loss: %.8f", epoch + 1,
                            train_loss / num, cluster_loss_val / num, recon_loss_val / num)

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
