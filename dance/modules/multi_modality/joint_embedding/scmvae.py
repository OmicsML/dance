"""Reimplementation of scMVAE model.

Extended from https://github.com/cmzuo11/scMVAE

Reference
---------
Chunman Zuo, Luonan Chen. Deep-joint-learning analysis model of single cell transcriptome and open chromatin accessibility data. Briefings in Bioinformatics. 2020.

"""
import collections
import copy
import math
import os
import time
import warnings
from copy import deepcopy

import numpy as np
import scipy.stats as stats
import torch
import torch.utils.data
import torch.utils.data as data_utils
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F
from tqdm import trange

from dance.utils.loss import GMM_loss
from dance.utils.metrics import integration_openproblems_evaluate

warnings.filterwarnings("ignore", category=DeprecationWarning)


def save_checkpoint(model, fileName='./saved_model/model_best.pth.tar'):
    folder = os.path.dirname(fileName)
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), fileName)


def load_checkpoint(file_path, model, device):
    model.load_state_dict(torch.load(file_path))
    model.to(device)

    return model


def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=1)


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    x = x.float()

    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log(theta + eps)

    log_theta_mu_eps = torch.log(theta + mu + eps)

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (-softplus_pi + pi_theta_log + x * (torch.log(mu + eps) - log_theta_mu_eps) +
                     torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))

    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return -torch.sum(res, dim=1)


def NB_loss(y_true, y_pred, theta, eps=1e-10):
    y_true = y_true.float()
    y_pred = y_pred.float()

    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred /
                                             (theta + eps))) + (y_true *
                                                                (torch.log(theta + eps) - torch.log(y_pred + eps)))

    final = t1 + t2

    return -torch.sum(final, dim=1)


def mse_loss(y_true, y_pred):
    mask = torch.sign(y_true)

    y_pred = y_pred.float()
    y_true = y_true.float()

    ret = torch.pow((y_pred - y_true) * mask, 2)

    return torch.sum(ret, dim=1)


def poisson_loss(y_true, y_pred):
    y_pred = y_pred.float()
    y_true = y_true.float()

    ret = y_pred - y_true * torch.log(y_pred + 1e-10) + torch.lgamma(y_true + 1.0)

    return torch.sum(ret, dim=1)


def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):
    lr = max(init_lr * (0.9**(iteration // adjust_epoch)), max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def build_multi_layers(layers, use_batch_norm=True, dropout_rate=0.1):
    """Build multilayer linear perceptron."""
    if dropout_rate > 0:
        fc_layers = nn.Sequential(
            collections.OrderedDict([(
                "Layer {}".format(i),
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                ),
            ) for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))]))

    else:
        fc_layers = nn.Sequential(
            collections.OrderedDict([(
                "Layer {}".format(i),
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                    nn.ReLU(),
                ),
            ) for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))]))

    return fc_layers


class Encoder(nn.Module):
    ## for one modulity
    def __init__(self, layer, hidden, Z_DIMS, dropout_rate=0.1):
        super().__init__()

        if len(layer) > 1:
            self.fc1 = build_multi_layers(layers=layer, dropout_rate=dropout_rate)

        self.layer = layer
        self.fc_means = nn.Linear(hidden, Z_DIMS)
        self.fc_logvar = nn.Linear(hidden, Z_DIMS)

    def reparametrize(self, means, logvar):

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(means)
        else:
            return means

    def forward(self, x):

        if len(self.layer) > 1:
            h = self.fc1(x)
        else:
            h = x
        mean_x = self.fc_means(h)
        logvar_x = self.fc_logvar(h)
        latent = self.reparametrize(mean_x, logvar_x)

        return mean_x, logvar_x, latent


class DecoderZINB(nn.Module):
    ### for scRNA-seq

    def __init__(self, layer, hidden, input_size, dropout_rate=0.1):

        super().__init__()

        if len(layer) > 1:
            self.decoder = build_multi_layers(layer, dropout_rate=dropout_rate)

        self.decoder_scale = nn.Linear(hidden, input_size)
        self.decoder_r = nn.Linear(hidden, input_size)
        self.dropout = nn.Linear(hidden, input_size)

        self.layer = layer

    def forward(self, z, library):

        if len(self.layer) > 1:
            latent = self.decoder(z)
        else:
            latent = z

        normalized_x = F.softmax(self.decoder_scale(latent), dim=1)  ## mean gamma

        recon_final = torch.exp(library) * normalized_x  ##mu
        disper_x = self.decoder_r(latent)  ### theta
        disper_x = torch.exp(disper_x)
        dropout_rate = self.dropout(latent)

        return dict(normalized=normalized_x, disperation=disper_x, imputation=recon_final, dropoutrate=dropout_rate)


class DecoderNB(nn.Module):

    ### for scRNA-seq

    def __init__(self, layer, hidden, input_size):
        super().__init__()

        self.decoder = build_multi_layers(layers=layer)

        self.decoder_scale = nn.Linear(hidden, input_size)
        self.decoder_r = nn.Linear(hidden, input_size)

    def forward(self, z, library):
        latent = self.decoder(z)

        normalized_x = F.softmax(self.decoder_scale(latent), dim=1)  ## mean gamma

        recon_final = torch.exp(library) * normalized_x  ##mu
        disper_x = self.decoder_r(latent)  ### theta
        disper_x = torch.exp(disper_x)

        return dict(normalized=normalized_x, disperation=disper_x, imputation=recon_final)


class Decoder(nn.Module):
    ### for scATAC-seq
    def __init__(self, layer, hidden, input_size, Type="Bernoulli", dropout_rate=0.1):
        super().__init__()

        if len(layer) > 1:
            self.decoder = build_multi_layers(layer, dropout_rate=dropout_rate)

        self.decoder_x = nn.Linear(hidden, input_size)
        self.Type = Type
        self.layer = layer

    def forward(self, z):

        if len(self.layer) > 1:
            latent = self.decoder(z)
        else:
            latent = z

        recon_x = self.decoder_x(latent)

        if self.Type == "Bernoulli":
            Final_x = torch.sigmoid(recon_x)

        elif self.Type == "Gaussian":
            Final_x = F.softmax(recon_x, dim=1)

        elif self.Type == "Gaussian1":
            Final_x = torch.sigmoid(recon_x)

        else:
            Final_x = F.relu(recon_x)

        return Final_x


class scMVAE(nn.Module):
    ## scMVAE-PoE

    def __init__(self, encoder_1, hidden_1, Z_DIMS, decoder_share, share_hidden, decoder_1, hidden_2, encoder_l,
                 hidden3, encoder_2, hidden_4, encoder_l1, hidden3_1, decoder_2, hidden_5, drop_rate,
                 log_variational=True, Type='Bernoulli', device='cpu', n_centroids=19, penality="GMM", model=2):

        super().__init__()

        self.X1_encoder = Encoder(encoder_1, hidden_1, Z_DIMS, dropout_rate=drop_rate)
        self.X1_encoder_l = Encoder(encoder_l, hidden3, 1, dropout_rate=drop_rate)

        self.X1_decoder = DecoderZINB(decoder_1, hidden_2, encoder_1[0], dropout_rate=drop_rate)

        self.X2_encoder = Encoder(encoder_2, hidden_4, Z_DIMS, dropout_rate=drop_rate)

        self.decode_share = build_multi_layers(decoder_share, dropout_rate=drop_rate)

        if Type == 'ZINB':
            self.X2_encoder_l = Encoder(encoder_l1, hidden3_1, 1, dropout_rate=drop_rate)
            self.decoder_x2 = DecoderZINB(decoder_2, hidden_5, encoder_2[0], dropout_rate=drop_rate)
        elif Type == 'Bernoulli':
            self.decoder_x2 = Decoder(decoder_2, hidden_5, encoder_2[0], Type, dropout_rate=drop_rate)
        elif Type == "Possion":
            self.decoder_x2 = Decoder(decoder_2, hidden_5, encoder_2[0], Type, dropout_rate=drop_rate)
        else:
            self.decoder_x2 = Decoder(decoder_2, hidden_5, encoder_2[0], Type, dropout_rate=drop_rate)

        self.experts = ProductOfExperts()
        self.Z_DIMS = Z_DIMS
        self.share_hidden = share_hidden
        self.log_variational = log_variational
        self.Type = Type
        self.decoder_share = decoder_share
        self.decoder_1 = decoder_1
        self.n_centroids = n_centroids
        self.penality = penality
        self.device = device
        self.model = model

        self.pi = nn.Parameter(torch.ones(n_centroids) / n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(Z_DIMS, n_centroids))  # mu
        self.var_c = nn.Parameter(torch.ones(Z_DIMS, n_centroids))  # sigma^2

    def _reparametrize(self, means, logvar):

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(means)
        else:
            return means

    def _encode_modalities(self, X1=None, X2=None):

        if X1 is not None:
            batch_size = X1.size(0)
        else:
            batch_size = X2.size(0)

        # Initialization
        means, logvar = prior_expert((1, batch_size, self.Z_DIMS))

        means = means.to(self.device)
        logvar = logvar.to(self.device)

        # Support for weak supervision setting
        if X1 is not None:
            X1_mean, X1_logvar, _ = self.X1_encoder(X1)
            means = torch.cat((means, X1_mean.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, X1_logvar.unsqueeze(0)), dim=0)

        if X2 is not None:
            X2_mean, X2_logvar, _ = self.X2_encoder(X2)
            means = torch.cat((means, X2_mean.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, X2_logvar.unsqueeze(0)), dim=0)

        # Combine the gaussians
        means, logvar = self.experts(means, logvar)
        return means, logvar

    def _inference(self, X1=None, X2=None):

        X1_ = X1
        X2_ = X2
        ### X1 processing
        mean_l, logvar_l, library = None, None, None

        if X1 is not None:
            if self.log_variational:
                X1_ = torch.log(X1_ + 1)

            mean_l, logvar_l, library = self.X1_encoder_l(X1_)

        ### X2 processing
        mean_l2, logvar_l2, library2 = None, None, None

        if X2 is not None:

            if self.Type == 'ZINB':
                if self.log_variational:
                    X2_ = torch.log(X2_ + 1)
                    mean_l2, logvar_l2, library2 = self.X2_encoder_l(X2_)

        means, logvar = self._encode_modalities(X1_, X2_)

        z = self._reparametrize(means, logvar)

        if len(self.decoder_share) > 1:
            latents = self.decode_share(z)

            if self.model == 0:
                latent_1 = latents
                latent_2 = latents

            elif self.model == 1:
                latent_1 = latents[:, :self.share_hidden]
                latent_2 = latents[:, self.share_hidden:]

            elif self.model == 2:
                latent_1 = torch.cat((z, latents[:, :self.share_hidden]), 1)
                latent_2 = latents[:, self.share_hidden:]

            else:
                latent_1 = torch.cat((z, latents), 1)
                latent_2 = latents
        else:
            latent_1 = z
            latent_2 = z

        # Reconstruct
        output = self.X1_decoder(latent_1, library)

        normalized_x = output["normalized"]
        recon_X1 = output["imputation"]
        disper_x = output["disperation"]
        dropout_rate = output["dropoutrate"]

        if self.Type == 'ZINB':
            results = self.decoder_x2(latent_2, library2)

            norma_x2 = results["normalized"]
            recon_X2 = results["imputation"]
            disper_x2 = results["disperation"]
            dropout_rate_2 = results["dropoutrate"]

        else:
            recon_X2 = self.decoder_x2(latent_2)
            norma_x2, disper_x2, dropout_rate_2 = None, None, None

        return dict(
            norm_x1=normalized_x,
            disper_x=disper_x,
            recon_x1=recon_X1,
            dropout_rate=dropout_rate,
            norm_x2=norma_x2,
            disper_x2=disper_x2,
            recon_x_2=recon_X2,
            dropout_rate_2=dropout_rate_2,
            mean_l=mean_l,
            logvar_l=logvar_l,
            library=library,
            mean_l2=mean_l2,
            logvar_l2=logvar_l2,
            library2=library2,
            mean_z=means,
            logvar_z=logvar,
            latent_z=z,
        )

    def forward(self, X1, X2, local_l_mean, local_l_var, local_l_mean1, local_l_var1):
        """Forward function for torch.nn.Module. An alias of encode_Batch function.

        Parameters
        ----------
        X1 : torch.utils.data.DataLoader
            Dataloader for dataset.
        X2 :
        local_l_mean:
        local_l_var:

        Returns
        -------
        latent_z1 : numpy.ndarray
            Latent representation of modality 1.
        latent_z2 : numpy.ndarray
            Latent representation of modality 2.
        norm_x1 : numpy.ndarray
            Normalized representation of modality 1.
        recon_x1 : numpy.ndarray
            Reconstruction result of modality 1.
        norm_x2 : numpy.ndarray
            Normalized representation of modality 2.
        recon_x2 : numpy.ndarray
            Reconstruction result of modality 2.

        """
        result = self._inference(X1, X2)

        disper_x = result["disper_x"]
        recon_x1 = result["recon_x1"]
        dropout_rate = result["dropout_rate"]

        disper_x2 = result["disper_x2"]
        recon_x_2 = result["recon_x_2"]
        dropout_rate_2 = result["dropout_rate_2"]

        if X1 is not None:
            mean_l = result["mean_l"]
            logvar_l = result["logvar_l"]

            kl_divergence_l = kl(Normal(mean_l, torch.exp(logvar_l)), Normal(local_l_mean,
                                                                             torch.sqrt(local_l_var))).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0)

        if X2 is not None:
            if self.Type == 'ZINB':
                mean_l2 = result["mean_l2"]
                logvar_l2 = result["library2"]
                kl_divergence_l2 = kl(Normal(mean_l2, torch.exp(logvar_l2)),
                                      Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)
            else:
                kl_divergence_l2 = torch.tensor(0.0)
        else:
            kl_divergence_l2 = torch.tensor(0.0)

        mean_z = result["mean_z"]
        logvar_z = result["logvar_z"]
        latent_z = result["latent_z"]

        if self.penality == "GMM":
            gamma, mu_c, var_c, pi = self._get_gamma(latent_z)  # , self.n_centroids, c_params)
            kl_divergence_z = GMM_loss(gamma, (mu_c, var_c, pi), (mean_z, torch.exp(logvar_z)))

        else:
            mean = torch.zeros_like(mean_z)
            scale = torch.ones_like(logvar_z)
            kl_divergence_z = kl(Normal(mean_z, torch.exp(logvar_z)), Normal(mean, scale)).sum(dim=1)

        loss1, loss2 = get_both_recon_loss(X1, recon_x1, disper_x, dropout_rate, X2, recon_x_2, disper_x2,
                                           dropout_rate_2, "ZINB", self.Type)

        return loss1, loss2, kl_divergence_l, kl_divergence_l2, kl_divergence_z

    def _out_Batch(self, Dataloader, out='Z', transforms=None):
        output = []

        for i, (X1, X2) in enumerate(Dataloader):

            X1 = X1.view(X1.size(0), -1).float().to(self.device)
            X2 = X2.view(X2.size(0), -1).float().to(self.device)

            result = self._inference(X1, X2)

            if out == 'Z':
                output.append(result["latent_z"].detach().cpu())
            elif out == 'recon_X1':
                output.append(result["recon_x1"].detach().cpu().data)
            elif out == 'Norm_X1':
                output.append(result["norm_x1"].detach().cpu().data)
            elif out == 'recon_X2':
                output.append(result["recon_x_2"].detach().cpu().data)

            elif out == 'logit':
                print('Logit not supported.')
                # output.append(self.get_gamma(z)[0].cpu().detach())

        output = torch.cat(output).numpy()

        return output

    def _get_gamma(self, z):

        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1)  # NxK
        mu_c = self.mu_c.repeat(N, 1, 1)  # NxDxK
        var_c = self.var_c.repeat(N, 1, 1)  # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(
            torch.log(pi) - torch.sum(0.5 * torch.log(2 * math.pi * var_c) + (z - mu_c)**2 /
                                      (2 * var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, Dataloader):
        """This function will initialize the parameters for PoE model.

        Parameters
        ----------
        Dataloader : torch.utils.data.DataLoader
            Dataloader for the whole dataset.

        Returns
        -------
        None.

        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')

        latent_z = self._out_Batch(Dataloader, out='Z')
        gmm.fit(latent_z)

        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def _denoise_batch(self, total_loader):
        # processing large-scale datasets
        latent_z = []
        norm_x1 = []
        recon_x_2 = []
        recon_x1 = []
        norm_x2 = []

        for batch_idx, (X1, X2) in enumerate(total_loader):
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)

            X1 = Variable(X1)
            X2 = Variable(X2)

            result = self._inference(X1, X2)

            latent_z.append(result["latent_z"].data.cpu().numpy())
            recon_x_2.append(result["recon_x_2"].data.cpu().numpy())
            recon_x1.append(result["recon_x1"].data.cpu().numpy())
            norm_x1.append(result["norm_x1"].data.cpu().numpy())
            norm_x2.append(result["norm_x2"].data.cpu().numpy())

        latent_z = np.concatenate(latent_z)
        recon_x_2 = np.concatenate(recon_x_2)
        recon_x1 = np.concatenate(recon_x1)
        norm_x1 = np.concatenate(norm_x1)
        norm_x2 = np.concatenate(norm_x2)

        return latent_z, recon_x1, norm_x1, recon_x_2, norm_x2

    def fit(self, args, train, valid, final_rate, scale_factor, device):
        """Fit function for training.

        Parameters
        ----------
        train : torch.utils.data.DataLoader
            Dataloader for training dataset.
        valid : torch.utils.data.DataLoader
            Dataloader for testing dataset.
        final_rate : torch.utils.data.DataLoader
            Dataloader for both training and testing dataset, for extra evaluation purpose.
        scale_factor : str
            Type of modality 1.
        device : torch.device

        Returns
        -------
        None.

        """
        train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)
        test_loader = data_utils.DataLoader(valid, batch_size=len(valid), shuffle=False)

        # args.max_epoch = 500
        # args.epoch_per_test = 10
        train_loss_list = []

        flag_break = 0
        epoch_count = 0
        reco_epoch_test = 0
        test_like_max = 100000

        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)

        epoch = 0
        iteration = 0
        start = time.time()

        with trange(args.max_epoch, disable=True) as pbar:
            while True:
                self.train()

                epoch += 1
                epoch_lr = adjust_learning_rate(args.lr, optimizer, epoch, final_rate, 10)
                kl_weight = min(1, epoch / args.anneal_epoch)

                for batch_idx, (X1, lib_m1, lib_v1, lib_m2, lib_v2, X2) in enumerate(train_loader):
                    X1, X2 = X1.float().to(device), X2.float().to(device)
                    lib_m1, lib_v1 = lib_m1.to(device), lib_v1.to(device)
                    lib_m2, lib_v2 = lib_m2.to(device), lib_v2.to(device)

                    X1, X2 = Variable(X1), Variable(X2)
                    lib_m1, lib_v1 = Variable(lib_m1), Variable(lib_v1)
                    lib_m2, lib_v2 = Variable(lib_m2), Variable(lib_v2)

                    optimizer.zero_grad()

                    loss1, loss2, kl_divergence_l, kl_divergence_l1, kl_divergence_z = self(
                        X1.float(), X2.float(), lib_m1, lib_v1, lib_m2, lib_v2)
                    loss = torch.mean((scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1) +
                                      (kl_weight * (kl_divergence_z)))

                    loss.backward()
                    optimizer.step()

                    iteration += 1

                epoch_count += 1

                if epoch % args.epoch_per_test == 0 and epoch > 0:

                    self.eval()

                    with torch.no_grad():

                        for batch_idx, (X1, lib_m1, lib_v1, lib_m2, lib_v2, X2) in enumerate(test_loader):

                            X1, X2 = X1.float().to(device), X2.float().to(device)
                            lib_v1, lib_m1 = lib_v1.to(device), lib_m1.to(device)
                            lib_v2, lib_m2 = lib_v2.to(device), lib_m2.to(device)

                            X1, X2 = Variable(X1), Variable(X2)
                            lib_m1, lib_v1 = Variable(lib_m1), Variable(lib_v1)
                            lib_m2, lib_v2 = Variable(lib_m2), Variable(lib_v2)

                            loss1, loss2, kl_divergence_l, kl_divergence_l1, kl_divergence_z = self(
                                X1.float(), X2.float(), lib_m1, lib_v1, lib_m2, lib_v2)
                            test_loss = torch.mean((scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1) +
                                                   (kl_weight * (kl_divergence_z)))

                            train_loss_list.append(test_loss.item())

                            if math.isnan(test_loss.item()):
                                flag_break = 1
                                break

                            if test_like_max > test_loss.item():
                                best_dict = deepcopy(self.state_dict())
                                test_like_max = test_loss.item()
                                epoch_count = 0
                                best_dict = deepcopy(self.state_dict())

                                print(
                                    str(epoch) + "   " + str(loss.item()) + "   " + str(test_loss.item()) + "   " +
                                    str(torch.mean(loss1).item()) + "   " + str(torch.mean(loss2).item()) +
                                    "  kl_divergence_l:  " + str(torch.mean(kl_divergence_l).item()) + " kl_weight: " +
                                    str(kl_weight) + " kl_divergence_z: " + str(torch.mean(kl_divergence_z).item()))

                if flag_break == 1:
                    reco_epoch_test = epoch
                    status = " With NA, training failed. "
                    break

                if epoch >= args.max_epoch:
                    reco_epoch_test = epoch
                    status = f" Reached {args.max_epoch} epoch, training complete. "
                    break

                if len(train_loss_list) >= 2:
                    if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-4:
                        reco_epoch_test = epoch
                        status = " Training for the train dataset is converged! "
                        break

        duration = time.time() - start
        print('Finish training, total time: ' + str(duration) + 's' + " epoch: " + str(reco_epoch_test) + " status: " +
              status)
        self.load_state_dict(best_dict)


#         load_checkpoint('./saved_model/model_best.pth.tar', self, device)

    def predict(self, X1, X2, out='Z', device='cpu'):
        """Predict function to get prediction.

        Parameters
        ----------
        X1 : torch.Tensor
            Features of modality 1.
        X2 : torch.Tensor
            Features of modality 2.
        out : str optional
            The ground truth labels for evaluation.

        Returns
        -------
        result : torch.Tensor
            The requested result, by default to be embedding in latent space (a.k.a 'Z').

        """
        with torch.no_grad():
            X1 = X1.view(X1.size(0), -1).float()
            X2 = X2.view(X2.size(0), -1).float()
            if device == 'cpu':
                model = copy.deepcopy(self).to('cpu')
                model.device = 'cpu'
                X1 = X1.to('cpu')
                X2 = X2.to('cpu')
                result = model._inference(X1, X2)
            else:
                X1 = X1.to(self.device)
                X2 = X2.to(self.device)
                result = self._inference(X1, X2)

            if out == 'Z':
                return result["latent_z"]
            elif out == 'recon_X1':
                return result["recon_x1"]
            elif out == 'Norm_X1':
                return result["norm_x1"]
            elif out == 'recon_X2':
                return result["recon_x_2"]
            elif out == 'logit':
                print('Logit not supported.')
                # output.append(self.get_gamma(z)[0].cpu().detach())
                return None

    def score(self, X1, X2, labels, adata_sol=None, metric='clustering'):
        """Score function to get score of prediction.

        Parameters
        ----------
        X1 : torch.Tensor
            Features of modality 1.
        X2 : torch.Tensor
            Features of modality 2.
        labels : torch.Tensor
            The ground truth labels for evaluation.

        Returns
        -------
        NMI_score : float
            NMI eval score.
        ARI_score : float
            ARI eval score.

        """

        if metric == 'clustering':
            emb = self.predict(X1, X2).cpu().numpy()
            kmeans = KMeans(n_clusters=10, n_init=5, random_state=200)

            true_labels = labels.numpy()
            pred_labels = kmeans.fit_predict(emb)

            NMI_score = round(normalized_mutual_info_score(true_labels, pred_labels, average_method='max'), 3)
            ARI_score = round(adjusted_rand_score(true_labels, pred_labels), 3)

            return {'dance_nmi': NMI_score, 'dance_ari': ARI_score}
        elif metric == 'openproblems':
            emb = self.predict(X1, X2).cpu().numpy()
            assert adata_sol, 'adata_sol is required by `openproblems` evaluation but not provided.'
            adata_sol.obsm['X_emb'] = emb
            return integration_openproblems_evaluate(adata_sol)
        else:
            raise NotImplementedError


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts. See
    https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts @param logvar: M x D for M experts

    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


def prior_expert(size):
    """Universal prior expert. Here we use a spherical.

    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian

    """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))

    return mu, logvar


def get_triple_recon_loss(x1=None, px1_rate=None, px1_r=None, px1_dropout=None, x2=None, px2_rate=None, px2_r=None,
                          px2_dropout=None, px12_rate=None, px12_r=None, px12_dropout=None, Type1="ZINB",
                          Type="Bernoulli"):
    reconst_loss1 = log_zinb_positive(x1, px1_rate, px1_r, px1_dropout)
    reconst_loss12 = binary_cross_entropy(px12_rate, x2)

    if x2 is not None:
        reconst_loss2 = binary_cross_entropy(px2_rate, x2)
    else:
        reconst_loss2 = torch.tensor(0.0)

    return reconst_loss1, reconst_loss2, reconst_loss12


def get_both_recon_loss(x1=None, px1_rate=None, px1_r=None, px1_dropout=None, x2=None, px2_rate=None, px2_r=None,
                        px2_dropout=None, Type1="ZINB", Type="Bernoulli"):
    # Reconstruction Loss
    ## here Type1 for rna-seq, Type for atac-seq
    # reconst_loss1 = log_zinb_positive( x1, px1_rate, px1_r, px1_dropout )
    if x1 is not None:
        reconst_loss1 = log_zinb_positive(x1, px1_rate, px1_r, px1_dropout)
    else:
        reconst_loss1 = torch.tensor(0.0)

    if x2 is not None:

        if Type == "ZINB":
            reconst_loss2 = log_zinb_positive(x2, px2_rate, px2_r, px2_dropout)

        elif Type == 'Bernoulli':
            reconst_loss2 = binary_cross_entropy(px2_rate, x2)

        elif Type == "Possion":
            reconst_loss2 = poisson_loss(x2, px2_rate)

        else:
            reconst_loss2 = mse_loss(x2, px2_rate)
    else:
        reconst_loss2 = torch.tensor(0.0)

    return reconst_loss1, reconst_loss2
