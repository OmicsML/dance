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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import TAGConv
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.nn import Parameter
from tqdm import tqdm

from dance.utils.loss import ZINBLoss, dist_loss
from dance.utils.metrics import cluster_acc


class SCTAG(nn.Module):
    """scTAG class.

    Parameters
    ----------
    X :
        input features.
    adj :
        adjacency matrix.
    adj_n :
        normalized adjacency matrix.
    n_clusters : int
        number of clusters.
    k : int optional
        number of hops of TAG convolutional layer.
    hidden_dim : int optional
        dimension of hidden layer.
    latent_dim : int optional
        dimension of latent embedding.
    dec_dim : list optional
        dimensions of decoder layers.
    adj_dim : int optional
        dimension of adjacency matrix.
    dropout : float optional
        dropout rate.
    device : str optional
        computing device.
    alpha : float optional
        parameter of soft assign.

    """

    def __init__(self, X, adj, adj_n, n_clusters, k=3, hidden_dim=128, latent_dim=15, dec_dim=None, adj_dim=32,
                 dropout=0.2, device="cuda", alpha=1.0):
        super().__init__()
        if dec_dim is None:
            dec_dim = [128, 256, 512]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.adj = adj
        self.adj_n = adj_n
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.device = device
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.k = k
        self.adj_dim = adj_dim
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.latent_dim).to(self.device))

        src, dist = np.nonzero(adj)
        self.G = dgl.graph((src, dist)).to(device)

        src_n, dist_n = np.nonzero(adj_n)
        self.G_n = dgl.graph((src_n, dist_n)).to(device)

        self.encoder1 = TAGConv(self.in_dim, self.hidden_dim, k=k)
        self.encoder2 = TAGConv(self.hidden_dim, self.latent_dim, k=k)
        self.decoderA = DecoderA(latent_dim=self.latent_dim, adj_dim=self.adj_dim, activation=torch.sigmoid,
                                 dropout=self.dropout)
        self.decoderX = DecoderX(self.in_dim, self.latent_dim, n_dec_1=dec_dim[0], n_dec_2=dec_dim[1],
                                 n_dec_3=dec_dim[2])
        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(self.device)

    def forward(self, A_in, X_input):
        """Forward propagation.

        Parameters
        ----------
        A_in :
            input adjacency matrix.
        X_input :
            input features.

        Returns
        -------
        A_out :
            reconstructed adjacency matrix.
        z :
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
        enc_h = self.encoder1(A_in, X_input)
        z = self.encoder2(A_in, enc_h)
        A_out = self.decoderA(z)
        _mean, _disp, _pi = self.decoderX(z)
        q = self.soft_assign(z)

        return A_out, z, q, _mean, _disp, _pi

    def pre_train(self, x, x_raw, fname, scale_factor, epochs=1000, info_step=10, lr=5e-4, W_a=0.3, W_x=1, W_d=0,
                  min_dist=0.5, max_dist=20.):
        """Pretrain autoencoder.

        Parameters
        ----------
        x :
            input features.
        x_raw :
            raw input features.
        fname : str
            path to save autoencoder weights.
        scale_factor : list
            scale factor of input features and raw input features.
        epochs : int optional
            number of epochs.
        info_step : int optional
            interval of showing pretraining loss.
        lr : float optional
            learning rate.
        W_a : float optional
            parameter of reconstruction loss.
        W_x : float optional
            parameter of ZINB loss.
        W_d : float optional
            parameter of pairwise distance loss.
        min_dist : float optional
            minimum distance of pairwise distance loss.
        max_dist : float optional
            maximum distance of pairwise distance loss.

        Returns
        -------
        None.

        """
        x = torch.Tensor(x).to(self.device)
        x_raw = torch.Tensor(x_raw).to(self.device)
        scale_factor = torch.tensor(scale_factor).to(self.device)

        self.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        print("Pretraining stage")
        for epoch in range(epochs):
            A_out, z, _, mean, disp, pi = self.forward(self.G_n, x)

            if W_d:
                Dist_loss = torch.mean(dist_loss(z, min_dist, max_dist=max_dist))
            A_rec_loss = torch.mean(F.mse_loss(A_out, torch.Tensor(self.adj).to(self.device)))
            Zinb_loss = self.zinb_loss(x_raw, mean, disp, pi, scale_factor)
            loss = W_a * A_rec_loss + W_x * Zinb_loss
            if W_d:
                loss += W_d * Dist_loss

            if epoch % info_step == 0:
                if W_d:
                    print("Epoch %3d: ZINB Loss: %.8f, MSE Loss: %.8f, Dist Loss: %.8f" %
                          (epoch + 1, Zinb_loss.item(), A_rec_loss.item(), Dist_loss.item()))
                else:
                    print("Epoch %3d: ZINB Loss: %.8f, MSE Loss: %.8f" %
                          (epoch + 1, Zinb_loss.item(), A_rec_loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(self.state_dict(), fname)
        print("Pre_train Finish!")

    def fit(self, x, x_raw, y, scale_factor, epochs=300, lr=5e-4, W_a=0.3, W_x=1, W_c=1.5, info_step=1):
        """Pretrain autoencoder.

        Parameters
        ----------
        x :
            input features.
        x_raw :
            raw input features.
        y : list
            true label.
        scale_factor : list
            scale factor of input features and raw input features.
        epochs : int optional
            number of epochs.
        lr : float optional
            learning rate.
        W_a : float optional
            parameter of reconstruction loss.
        W_x : float optional
            parameter of ZINB loss.
        W_c : float optional
            parameter of clustering loss.
        info_step : int optional
            interval of showing pretraining loss.

        Returns
        -------
        None.

        """
        x = torch.Tensor(x).to(self.device)
        x_raw = torch.Tensor(x_raw).to(self.device)
        scale_factor = torch.tensor(scale_factor).to(self.device)

        # Initializing cluster centers with kmeans
        kmeans = KMeans(self.n_clusters, n_init=20)
        enc_h = self.encoder1(self.G_n, x)
        z = self.encoder2(self.G_n, enc_h)
        kmeans.fit_predict(z.detach().cpu().numpy())
        self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device))

        self.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)

        aris = []
        P = {}
        Q = {}

        for epoch in range(epochs):
            A_out, _, q, mean, disp, pi = self.forward(self.G_n, x)
            self.q = q
            p = self.target_distribution(q)
            self.y_pred = self.predict()

            # calculate ari score for model selection
            _, _, ari = self.score(y)
            aris.append(ari)
            p_ = {f'epoch{epoch}': p}
            q_ = {f'epoch{epoch}': q}
            P = {**P, **p_}
            Q = {**Q, **q_}
            if epoch % info_step == 0:
                print("Epoch %3d, ARI: %.4f, Best ARI: %.4f" % (epoch + 1, ari, max(aris)))

            A_rec_loss = torch.mean(F.mse_loss(A_out, torch.Tensor(self.adj).to(self.device)))
            Zinb_loss = self.zinb_loss(x_raw, mean, disp, pi, scale_factor)
            Cluster_loss = torch.mean(
                F.kl_div(
                    torch.Tensor(self.y_pred).to(self.device),
                    torch.Tensor(y).to(self.device), reduction='batchmean'))
            loss = W_a * A_rec_loss + W_x * Zinb_loss + W_c * Cluster_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        index = np.argmax(aris)
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


class DecoderA(nn.Module):
    """Decoder class for adjacency matrix reconstruction.

    Parameters
    ----------
    latent_dim : int optional
        dimension of latent embedding.
    adj_dim : int optional
        dimension of adjacency matrix.
    activation : optional
        activation function.
    dropout : float optional
        dropout rate.

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
        z :
            embedding.

        Returns
        -------
        adj :
            reconstructed adjacency matrix.

        """
        dec_h = self.dec_1(z)
        z0 = F.dropout(dec_h, self.dropout)
        adj = self.activation(torch.mm(z0, z0.t()))
        return adj


class DecoderX(nn.Module):
    """scTAG class.

    Parameters
    ----------
    input_dim : int
        dimension of input feature.
    n_z : int
        dimension of latent embedding.
    n_dec_1 : int optional
        number of nodes of decoder layer 1.
    n_dec_2 : int optional
        number of nodes of decoder layer 2.
    n_dec_3 : int optional
        number of nodes of decoder layer 3.

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
        z :
            embedding.

        Returns
        -------
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB.

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
