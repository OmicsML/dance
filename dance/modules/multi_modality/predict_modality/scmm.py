"""Reimplementation of scMM method.

Extended from https://github.com/kodaim1115/scMM

Reference
---------
Minoura, Kodai, et al. "A mixture-of-experts deep generative model for integrated analysis of single-cell multiomics data." Cell reports methods 1.5 (2021): 100071.

"""
import math
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from sklearn.cluster import DBSCAN, KMeans
from torch import optim
from torch.utils.data import DataLoader

from dance.utils import SimpleIndexDataset


def get_mean(d, K=100):
    """Extract the `mean` parameter for given distribution.

    If attribute not available, estimate from samples.

    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def m_elbo_naive(model, x):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED."""
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model._get_pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            lpx_zs.append(lpx_z.sum(-1))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.sum()


def m_elbo_naive_warmup(model, x, beta):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED."""
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model._get_pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            lpx_zs.append(lpx_z.sum(-1))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - beta * torch.stack(klds).sum(0))
    return obj.sum()


def protein_preprocessing(t1):
    t0 = t1.clone()
    t0[t0 == 0] = 1
    return torch.log1p(t1 / torch.exp(torch.sum(torch.log(t0), axis=1) * (1 / torch.sum(t1 > 0, axis=1))).unsqueeze(-1))


def atac_preprocessing(t1):
    t1[t1 > 0] = 1
    return t1


# TODO: Not implemented
def rna_preprocessing(t1):
    return t1


class Constants:
    eta = 1e-6
    eps = 1e-7
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


class ZINB(ZeroInflatedNegativeBinomial):

    def __init__(self, total_count, probs, gate):
        super().__init__(total_count=total_count, probs=probs, gate=gate)


class VAE(nn.Module):

    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params):
        super().__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self._pz_params = None  # defined in subclass
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

    @property
    def pz_params(self):
        return self._pz_params

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample()
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample()  # no dim expansion
            px_z = self.px_z(*self.dec(latents))
            recon = get_mean(px_z)
        return recon

    def reconstruct_sample(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample()  # no dim expansion
            px_z = self.px_z(*self.dec(latents))
            recon = px_z._sample()
        return recon

    def latents(self, data, sampling=False):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            if not sampling:
                lats = get_mean(qz_x)
            else:
                lats = qz_x._sample()
        return lats


class Enc(nn.Module):

    def __init__(self, data_dim, latent_dim, num_hidden_layers, hidden_dim):  # added hidden_dim
        super().__init__()
        self.data_dim = data_dim
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        for _ in range(num_hidden_layers - 1):
            modules.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.scale_factor = 10000

    def read_count(self, x):
        read = torch.sum(x, axis=1)
        read = read.repeat(self.data_dim, 1).t()
        return (read)

    def forward(self, x):
        read = self.read_count(x)
        x = x / read * self.scale_factor
        e = self.enc(x)
        lv = self.fc22(e).clamp(-12, 12)  # restrict to avoid torch.exp() over/underflow
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Dec(nn.Module):
    """Generate an MNIST image given a sample from the latent space."""

    def __init__(self, data_dim, latent_dim, num_hidden_layers, hidden_dim, modality):  # added hidden_dim
        super().__init__()
        self.modality = modality
        self.data_dim = data_dim

        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        for _ in range(num_hidden_layers - 1):
            modules.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)
        self.fc32 = nn.Linear(hidden_dim, data_dim)

        if self.modality == 'atac':
            # zero-inflated
            self.fc33 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        d = self.dec(z)
        log_r = self.fc31(d).clamp(-12, 12)  # restrict to avoid torch.exp() over/underflow
        r = torch.exp(log_r)
        p = self.fc32(d)
        p = torch.sigmoid(p).clamp(Constants.eps, 1 - Constants.eps)  # restrict to avoid probs = 0,1

        if self.modality == 'atac':
            g = self.fc33(d)
            g = torch.sigmoid(g)
            return r, p, g
        else:
            return r, p


class ATAC(VAE):
    """Derive a specific sub-class of a VAE for ATAC."""

    def __init__(self, params):
        super().__init__(dist.Laplace, ZINB, dist.Laplace,
                         Enc(params.p_dim, params.latent_dim, params.num_hidden_layers, params.p_hidden_dim),
                         Dec(params.p_dim, params.latent_dim, params.num_hidden_layers, params.p_hidden_dim, 'atac'),
                         params)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'atac'
        self.data_dim = self.params.p_dim
        self.llik_scaling = 1.
        self.scale_factor = 10000

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(dataset, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)
        return dataloader

    def forward(self, x):
        read_count = self.enc.read_count(x)
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample()
        r, p, g = self.dec(zs)
        r = r / self.scale_factor * read_count
        px_z = self.px_z(r, p, g)
        return qz_x, px_z, zs


class Protein(VAE):
    """Derive a specific sub-class of a VAE for Protein."""

    def __init__(self, params):
        super().__init__(dist.Laplace, dist.NegativeBinomial, dist.Laplace,
                         Enc(params.p_dim, params.latent_dim, params.num_hidden_layers, params.p_hidden_dim),
                         Dec(params.p_dim, params.latent_dim, params.num_hidden_layers, params.p_hidden_dim, 'protein'),
                         params)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'protein'
        self.data_dim = self.params.p_dim
        self.llik_scaling = 1.
        self.scale_factor = 10000

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(dataset, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)

        return dataloader

    def forward(self, x):
        read_count = self.enc.read_count(x)
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample()
        r, _ = self.dec(zs)
        r = r / self.scale_factor * read_count
        px_z = self.px_z(r, _)
        return qz_x, px_z, zs


class RNA(VAE):
    """Derive a specific sub-class of a VAE for RNA."""

    def __init__(self, params):
        super().__init__(
            dist.Laplace,
            dist.NegativeBinomial,  # likelihood
            dist.Laplace,
            Enc(params.r_dim, params.latent_dim, params.num_hidden_layers, params.r_hidden_dim),
            Dec(params.r_dim, params.latent_dim, params.num_hidden_layers, params.r_hidden_dim, 'rna'),
            params)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'rna'
        self.data_dim = self.params.r_dim
        self.llik_scaling = 1.
        self.scale_factor = 10000

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(dataset, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)

        return dataloader

    def forward(self, x):
        read_count = self.enc.read_count(x)
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample()
        r, _ = self.dec(zs)
        r = r / self.scale_factor * read_count
        px_z = self.px_z(r, _)
        return qz_x, px_z, zs


class MMVAE(nn.Module):
    """MMVAE class.

    Parameters
    ----------
    subtask : str
        Name of the subtask which is composed of the name of two modality. This parameter will indicate some modality-specific features in the model.
    params : argparse.Namespace
        A Namespace object that contains arguments of MMVAE. For details of parameters in parser args, please refer to link (parser help document).

    """

    def __init__(self, subtask, params):
        super().__init__()
        self.pz = dist.Laplace
        assert subtask in ('rna-dna', 'rna-protein')
        self.modelName = subtask
        if subtask == 'rna-dna':
            self.preprocessing = atac_preprocessing
            self.vaes = nn.ModuleList([RNA(params), ATAC(params)])
        else:
            self.preprocessing = protein_preprocessing
            self.vaes = nn.ModuleList([RNA(params), Protein(params)])
        self.params = params

        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.scale_factor = 10000

    @property
    def _get_pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def _get_cluster(self, data, modality='both', n_clusters=10, method='kmeans', device='cuda'):
        self.eval()
        lats = self._get_latents(data, sampling=False)
        if modality == 'both':
            lat = sum(lats) / len(lats)
        elif modality == 'rna':
            lat = lats[0]
        elif modality == 'atac':
            lat = lats[1]

        if method == 'kmeans':
            fit = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(lat.cpu().numpy())
            cluster = fit.labels_
        elif method == 'dbscan':
            fit = DBSCAN(eps=0.5, min_samples=50).fit(lat.cpu().numpy())
            cluster = fit.labels_
        else:
            gamma, _, _, _, _ = self.get_gamma(lat)
            cluster = torch.argmax(gamma, axis=1)
            cluster = cluster.detach().numpy()
            fit = None

        return cluster, fit

    def forward(self, x):
        """Forward function for torch.nn.Module.

        Parameters
        ----------
        x : list[torch.Tensor]
            Features of two modalities.

        Returns
        -------
        qz_xs : list[torch.Tensor]
            Post prior of two modalities.
        px_zs : list[torch.Tensor]
            likelihood of two modalities.
        zss : list[torch.Tensor]
            Reconstruction results of two modalities.

        """
        qz_xs, zss = [], []
        read_counts = []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            read_counts.append(vae.enc.read_count(x[m]))
            qz_x, px_z, zs = vae(x[m])
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if self.modelName == 'rna-protein':
                        r, _ = vae.dec(zs)
                        r = r / self.scale_factor * read_counts[d]
                        px_zs[e][d] = vae.px_z(r, _)
                    else:
                        if d == 0:
                            r, p = vae.dec(zs)
                            r = r / self.scale_factor * read_counts[d]
                            px_zs[e][d] = vae.px_z(r, p)
                        else:
                            r, p, g = vae.dec(zs)
                            r = r / self.scale_factor * read_counts[d]
                            px_zs[e][d] = vae.px_z(r, p, g)

        return qz_xs, px_zs, zss

    def _reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions of MEANS
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons

    def _reconstruct_sample(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions of SAMPLES
            recons = [[px_z.sample() for px_z in r] for r in px_zs]
        return recons

    def _get_latents(self, data, sampling=False):
        self.eval()
        with torch.no_grad():
            qz_xs, _, _ = self.forward(data)
            if not sampling:
                lats = [get_mean(qz_x) for qz_x in qz_xs]
            else:
                lats = [qz_x._sample() for qz_x in qz_xs]
        return lats

    def fit(self, x_train, y_train, val_ratio=0.15):
        """Fit function for training.

        Parameters
        ----------
        x_train : torch.Tensor
            Input modality for training.

        y_train : torch.Tensor
            Target modality for training.

        val_ratio : float
            Ratio for automatic train-validation split.

        Returns
        -------
        None.

        """

        start_early_stop = self.params.deterministic_warmup

        idx = np.random.permutation(x_train.shape[0])
        train_idx = idx[:int(idx.shape[0] * (1 - val_ratio))]
        val_idx = idx[int(idx.shape[0] * (1 - val_ratio)):]

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.params.lr, amsgrad=True)
        assert (self.params.obj in ['m_elbo_naive', 'm_elbo_naive_warmup'])
        objective = m_elbo_naive_warmup if self.params.obj == 'm_elbo_naive_warmup' else 'm_elbo_naive'
        train_dataset = SimpleIndexDataset(train_idx)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        train_mod1 = x_train.float().to(self.params.device)
        train_mod2 = y_train.float().to(self.params.device)
        self.ratio = train_mod2.sum() / train_mod1.sum()
        vals = []
        tr = []

        try:
            for epoch in range(1, self.params.epochs + 1):
                self.train()
                b_loss = 0
                for i, batch_idx in enumerate(train_loader):
                    dataT = (train_mod1[batch_idx], train_mod2[batch_idx])
                    beta = (epoch - 1) / start_early_stop if epoch <= start_early_stop else 1
                    if dataT[0].size()[0] == 1:
                        continue
                    # data = [d.to(self.paradevice) for d in dataT]  # multimodal
                    data = dataT
                    optimizer.zero_grad()
                    loss = -objective(self, data, beta)
                    loss.backward()
                    optimizer.step()
                    b_loss += loss.item()
                    if self.params.print_freq > 0 and i % self.params.print_freq == 0:
                        print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / self.params.batch_size))
                tr.append(b_loss / len(train_loader.dataset))
                print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, tr[-1]))

                if torch.isnan(torch.tensor([b_loss])):
                    break

                vals.append(self.score(train_mod1[val_idx], train_mod2[val_idx], metric='loss'))
                print('====>             Valid loss: {:.4f}'.format(vals[-1]))

                if vals[-1] == min(vals):
                    if not os.path.exists('models'):
                        os.mkdir('models')

    #                 torch.save(self.state_dict(), f'models/model_{self.params.seed}.pth')
                    best_dict = deepcopy(self.state_dict())

                if epoch % 10 == 0:
                    print('Valid RMSELoss:', self.score(train_mod1[val_idx], train_mod2[val_idx]))

                if epoch > start_early_stop and min(vals) != min(vals[-10:]):
                    print('Early stopped.')
                    #                 self.load_state_dict(torch.load(f'models/model_{self.params.seed}.pth'))
                    break
        except:
            pass
        self.load_state_dict(best_dict)

    def score(self, X, Y, metric='rmse'):
        """Score function to get score of prediction.

        Parameters
        ----------
        X : torch.Tensor
            Features of input modality.
        Y : torch.Tensor
            Features of input modality.
        metric : str optional
            Metric of the score function, by default to be 'rmse'.

        Returns
        -------
        score : float
            Score of predicted matching, according to specified metric.

        """
        self.eval()
        self.eval()
        X = X.float().to(self.params.device)
        Y = Y.float().to(self.params.device)
        if metric == 'loss':
            b_loss = 0
            idx = np.arange(X.shape[0])
            dataset = SimpleIndexDataset(idx)
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            with torch.no_grad():
                for i, batch_idx in enumerate(data_loader):
                    objective = m_elbo_naive_warmup if self.params.obj == 'm_elbo_naive_warmup' else 'm_elbo_naive'
                    loss = -objective(self, [X[batch_idx], Y[batch_idx]], 1).item()
                    b_loss += loss
            return b_loss / X.shape[0]
        elif metric == 'rmse':
            mse = nn.MSELoss()
            pred_test = self.predict(X)
            pred = self.preprocessing(pred_test)
            pred = torch.nan_to_num(pred)
            label = self.preprocessing(Y)
            return math.sqrt(mse(pred, label).item())
        else:
            print('Warning: undefined evaluation metric.')

    def predict(self, X):
        """Score function to get score of prediction.

        Parameters
        ----------
        X : torch.Tensor
            Features of input modality and target modality.

        Returns
        -------
        pred : torch.Tensor
            Prediction of target modality from input modality.

        """
        self.eval()
        X = X.float().to(self.params.device)
        idx = np.arange(X.shape[0])
        dataset = SimpleIndexDataset(idx)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.params.batch_size * 10,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        # uni, cross = [], []
        pred = []
        with torch.no_grad():
            for i, batch_idx in enumerate(data_loader):
                read_count = self.vaes[1].enc.read_count(X[batch_idx] * self.ratio)
                qz_x, px_z, zs = self.vaes[0](X[batch_idx])
                if self.modelName == 'rna-protein':
                    r, _ = self.vaes[1].dec(zs)
                    r = r / self.scale_factor * read_count
                    pred.append(self.vaes[1].px_z(r, _).sample())
                else:
                    r, p, g = self.vaes[1].dec(zs)
                    r = r / self.scale_factor * read_count
                    pred.append(self.vaes[1].px_z(r, p, g).sample())
        pred = torch.cat(pred, 0)
        return pred

        #         recons_mat = self._reconstruct_sample(dataT)
        #         for e, recons_list in enumerate(recons_mat):
        #             for d, recon in enumerate(recons_list):
        #                 if e == d:
        #                     recon = recon.cpu()
        #                     if i == 0:
        #                         uni.append(recon)
        #                     else:
        #                         uni[e] = torch.cat([uni[e], recon])
        #                 else:
        #                     recon = recon.cpu()
        #                     if i == 0:
        #                         cross.append(recon)
        #                     else:
        #                         cross[e] = torch.cat((cross[e], recon))
        # return uni, cross
