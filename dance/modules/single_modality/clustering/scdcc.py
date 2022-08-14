"""Reimplementation of scDCC.

Extended from https://github.com/ttgump/scDCC

Reference
----------
Tian, Tian, et al. "Model-based deep embedding for constrained clustering analysis of single cell RNA-seq data."
Nature communications 12.1 (2021): 1-12.

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


class ScDCC(nn.Module):
    """scDCC class.

    Parameters
    ----------
    input_dim : int
        dimension of encoder input.
    z_dim : int
        dimension of embedding.
    n_clusters : int
        number of clusters.
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
    ml_weight : float optional
        parameter of must-link loss.
    cl_weight : float optional
        parameter of cannot-link loss.

    """

    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[], activation="relu", sigma=1.,
                 alpha=1., gamma=1., ml_weight=1., cl_weight=1.):
        super().__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().cpu()

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
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.to(device)

        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

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
        loss = self.gamma * kldloss
        return loss

    def pairwise_loss(self, p1, p2, cons_type):
        """Calculate pairwise loss.

        Parameters
        ----------
        p1 :
            distribution 1.
        p2 :
            distribution 2.
        cons_type : str
            type of loss.

        Returns
        -------
        loss :
            pairwise loss.

        """
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            loss = self.ml_weight * ml_loss
            return loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            loss = self.cl_weight * cl_loss
            return loss

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True,
                             ae_weights='AE_weights.pth.tar'):
        """Pretrain autoencoder.

        Parameters
        ----------
        x :
            input features.
        X_raw :
            raw input features.
        size_factor : list
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
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.to(device)
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(device)
                x_raw_tensor = Variable(x_raw_batch).to(device)
                sf_tensor = Variable(sf_batch).to(device)
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                      scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx + 1, epoch + 1, loss.item()))

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

    def fit(self, X, X_raw, sf, ml_ind1=np.array([]), ml_ind2=np.array([]), cl_ind1=np.array([]), cl_ind2=np.array([]),
            ml_p=1., cl_p=1., y=None, lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        """Train model.

        Parameters
        ----------
        X :
            input features.
        X_raw :
            raw input features.
        sf : float
            size factor of input features and raw input features.
        ml_ind1 : np.array optional
            index 1 of must-link pairs.
        ml_ind2 : np.array optional
            index 2 of must-link pairs.
        cl_ind1 : np.array optional
            index 1 of cannot-link pairs.
        cl_ind2 : np.array optional
            index 2 of cannot-link pairs.
        ml_p : float optional
            parameter of must-link loss.
        cl_p : float optional
            parameter of cannot-link loss.
        y : list optional
            true label. Used for model selection.
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

        print("Training stage")
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.to(device)
        X = torch.tensor(X).to(device)
        X_raw = torch.tensor(X_raw).to(device)
        sf = torch.tensor(sf).to(device)
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

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
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
                            'p': p,
                            'q': q,
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
                sfbatch = sf[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size:min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)
                target = Variable(pbatch)

                z, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss = cluster_loss + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val

            print("#Epoch %3d: Total: %.4f, Clustering Loss: %.4f, ZINB Loss: %.4f" %
                  (epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

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
                    inputs1 = Variable(px1)
                    rawinputs1 = Variable(pxraw1)
                    sfinput1 = Variable(sf1)
                    inputs2 = Variable(px2)
                    rawinputs2 = Variable(pxraw2)
                    sfinput2 = Variable(sf2)
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
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    z1, q1, _, _, _ = self.forward(inputs1)
                    z2, q2, _, _, _ = self.forward(inputs2)
                    loss = cl_p * self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch > 0 and cl_num_batch > 0:
                print("Pairwise Total: %.4f, ML loss: %.4f, CL loss: %.4f" %
                      (float(ml_loss.cpu()) + float(cl_loss.cpu()), ml_loss.cpu(), cl_loss.cpu()))

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
