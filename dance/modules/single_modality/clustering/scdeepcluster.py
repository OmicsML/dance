"""Reimplementation of scDeepCluster.

Extended from https://github.com/ttgump/scDeepCluster

Reference
----------
Tian, Tian, et al. "Clustering single-cell RNA-seq data with a model-based deep learning approach." Nature Machine
Intelligence 1.4 (2019): 191-198.

"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from dance.utils.loss import ZINBLoss
from dance.utils.metrics import cluster_acc


def buildNetwork(layers, type, activation="relu"):
    """Build network layer.

    Parameters
    ----------
    layers : list
        dimensions of layers.
    type : str
        type of network.
    activation : str optional
        activation function.

    Returns
    -------
    net :
        torch.nn network.

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


class ScDeepCluster(nn.Module):
    """scDeepCluster class.

    Parameters
    ----------
    input_dim : int
        dimension of encoder input.
    z_dim : int
        dimension of embedding.
    encodeLayer : list optional
        dimensions of encoder layers.
    decodeLayer : list optional
        dimensions of decoder layers.
    activation : str optional
        activation function.
    sigma : float optional
        parameter of Gaussian noise.
    alpha : float optional
        parameter of soft assign.
    gamma : float optional
        parameter of cluster loss.
    device : str optional
        computing device.

    """

    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], activation="relu", sigma=1., alpha=1.,
                 gamma=1., device="cuda"):
        super().__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)

    def save_model(self, path):
        """Save model to path.

        Parameters
        ----------
        path : str
            path to save model.

        Returns
        -------
        None.

        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model from path.

        Parameters
        ----------
        path : str
            path to load model.

        Returns
        -------
        None.

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
        z :
            embedding.

        Returns
        -------
        q :
            soft label.

        """
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        """Calculate auxiliary target distribution p with q.

        Parameters
        ----------
        q :
            soft label.

        Returns
        -------
        p :
            target distribution.

        """
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        """Forward propagation of autoencoder.

        Parameters
        ----------
        x :
            input features.

        Returns
        -------
        z0 :
            embedding.
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB.

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
        x :
            input features.

        Returns
        -------
        z0 :
            embedding.
        q :
            soft label.
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB.

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
        X :
            input features.
        batch_size : int optional
            size of batch.

        Returns
        -------
        encoded :
            embedding.

        """
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _, _ = self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)

    def cluster_loss(self, p, q):
        """Calculate cluster loss.

        Parameters
        ----------
        p :
            target distribution.
        q :
            soft label.

        Returns
        -------
        loss :
            cluster loss.

        """

        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)
        return self.gamma * kldloss

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True,
                             ae_weights='AE_weights.pth.tar'):
        """Pretrain autoencoder.

        Parameters
        ----------
        X :
            input features.
        X_raw :
            raw input features.
        size_factor : float
            size factor of input features and raw input features.
        batch_size : int optional
            size of batch.
        lr : float optional
            learning rate.
        epochs : int optional
            number of epochs.
        ae_save : bool optional
            save autoencoder weights or not.
        ae_weights : str optional
            path to save autoencoder weights.

        Returns
        -------
        None.

        """
        self.train()
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                      scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item() * len(x_batch)
            print('Pretrain epoch %3d, ZINB loss: %.8f' % (epoch + 1, loss_val / X.shape[0]))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        """Save training checkpoint.

        Parameters
        ----------
        state :
            model state
        index : int
            checkpoint index
        filename : str
            filename to save

        Returns
        -------
        None.

        """
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters, init_centroid=None, y=None, y_pred_init=None, lr=1.,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        """Train model.

        Parameters
        ----------
        X :
            input features.
        X_raw :
            raw input features.
        size_factor : list
            size factor of input features and raw input features.
        n_clusters : int
            number of clusters.
        init_centroid : list optional
            initialization of centroids. If None, perform kmeans to initialize cluster centers.
        y : list optional
            true label. Used for model selection.
        y_pred_init : list optional
            predicted label for initialization.
        lr : float optional
            learning rate.
        batch_size : int optional
            size of batch.
        num_epochs : int optional
            number of epochs.
        update_interval : int optional
            update interval of soft label and target distribution.
        tol : float optional
            tolerance for training loss.
        save_dir : str optional
            path to save model weights.

        Returns
        -------
        None.

        """
        self.train()
        print("Clustering stage")
        X = torch.tensor(X, dtype=torch.float32)
        X_raw = torch.tensor(X_raw, dtype=torch.float32)
        size_factor = torch.tensor(size_factor, dtype=torch.float32)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float32))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred

        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))

        aris = []
        P = {}
        Q = {}

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                self.q = q
                p = self.target_distribution(q).data
                self.y_pred = self.predict()

                # save current model
                if (epoch > 0 and delta_label < tol) or epoch % 10 == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                        }, epoch + 1, filename=save_dir)
                p_ = {f'epoch{epoch}': p}
                q_ = {f'epoch{epoch}': q}
                P = {**P, **p_}
                Q = {**Q, **q_}

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

                # calculate ari score for model selection
                _, _, ari = self.score(y)
                aris.append(ari)

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                xrawbatch = X_raw[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                sfbatch = size_factor[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                loss = cluster_loss * self.gamma + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                recon_loss_val += recon_loss.item() * len(inputs)
                train_loss += loss.item() * len(inputs)

            print("Epoch %3d: Total: %.8f, Clustering Loss: %.8f, ZINB Loss: %.8f" %
                  (epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

        index = update_interval * np.argmax(aris)
        self.q = Q[f'epoch{index}']

    def predict(self):
        """Get predictions from the trained model.

        Parameters
        ----------
        None.

        Returns
        -------
        y_pred : np.array
            prediction of given clustering method.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        return y_pred

    def score(self, y):
        """Evaluate the trained model.

        Parameters
        ----------
        y : list
            true labels.

        Returns
        -------
        acc : float
            accuracy.
        nmi : float
            normalized mutual information.
        ari : float
            adjusted Rand index.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        return acc, nmi, ari


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
