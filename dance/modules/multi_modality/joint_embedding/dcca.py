"""Reimplementation of the Deep Cross-omics Cycle Attention method.

Extended from https://github.com/cmzuo11/DCCA

Reference
---------
Chunman Zuo, Hao Dai, Luonan Chen. Deep cross-omics cycle attention model for joint analysis of single-cell multi-omics data. Bioinformatics. 2021.

"""

import collections
import math
import os
import time
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F

# from DCCA.loss_function import log_zinb_positive, log_nb_positive, binary_cross_entropy, mse_loss, KL_diver
from dance.utils.loss import Attention, Correlation, Eucli_dis, FactorTransfer, KL_diver, L1_dis, NSTLoss, Similarity

warnings.filterwarnings("ignore", category=DeprecationWarning)


def mse_loss(y_true, y_pred):
    mask = torch.sign(y_true)

    y_pred = y_pred.float()
    y_true = y_true.float()

    ret = torch.pow((y_pred - y_true) * mask, 2)

    return torch.sum(ret, dim=1)


def binary_cross_entropy(recon_x, x):
    # mask = torch.sign(x)
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


def log_nb_positive(x, mu, theta, eps=1e-8):
    x = x.float()

    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (theta * (torch.log(theta + eps) - log_theta_mu_eps) + x * (torch.log(mu + eps) - log_theta_mu_eps) +
           torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))

    return -torch.sum(res, dim=1)


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


def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):
    lr = max(init_lr * (0.9**(iteration // adjust_epoch)), max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


class Encoder(nn.Module):

    ## for one modulity
    def __init__(self, layer, hidden, Z_DIMS, droprate=0.1):
        super().__init__()

        if len(layer) > 1:
            self.fc1 = build_multi_layers(layers=layer, dropout_rate=droprate)

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

    def return_all_params(self, x):

        if len(self.layer) > 1:
            h = self.fc1(x)
        else:
            h = x

        mean_x = self.fc_means(h)
        logvar_x = self.fc_logvar(h)
        latent = self.reparametrize(mean_x, logvar_x)

        return mean_x, logvar_x, latent, h

    def forward(self, x):

        _, _, latent = self.return_all_params(x)

        return latent


class DecoderLogNormZINB(nn.Module):

    ### for scRNA-seq, refered by DCA

    def __init__(self, layer, hidden, input_size, droprate=0.1):
        super().__init__()

        self.decoder = build_multi_layers(layers=layer, dropout_rate=droprate)

        self.decoder_scale = nn.Linear(hidden, input_size)
        self.decoder_r = nn.Linear(hidden, input_size)
        self.dropout = nn.Linear(hidden, input_size)

    def forward(self, z=None, scale_factor=1.0):
        latent = self.decoder(z)

        normalized_x = F.softmax(self.decoder_scale(latent), dim=1)

        batch_size = normalized_x.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x.size(1))

        scale_x = torch.exp(scale_factor) * normalized_x  ###

        disper_x = torch.exp(self.decoder_r(latent))  ### theta
        dropout_rate = self.dropout(latent)

        return dict(normalized=normalized_x, disperation=disper_x, dropoutrate=dropout_rate, scale_x=scale_x)


class DecoderLogNormNB(nn.Module):

    ### for scRNA-seq

    def __init__(self, layer, hidden, input_size, droprate=0.1):
        super().__init__()

        self.decoder = build_multi_layers(layers=layer, dropout_rate=droprate)

        self.decoder_scale = nn.Linear(hidden, input_size)
        self.decoder_r = nn.Linear(hidden, input_size)

    def forward(self, z, scale_factor=torch.tensor(1.0)):
        latent = self.decoder(z)

        normalized_x = F.softmax(self.decoder_scale(latent), dim=1)  ## mean gamma

        batch_size = normalized_x.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x.size(1))

        scale_x = torch.exp(scale_factor) * normalized_x

        disper_x = torch.exp(self.decoder_r(latent))  ### theta

        return dict(
            normalized=normalized_x,
            disperation=disper_x,
            scale_x=scale_x,
        )


class Decoder(nn.Module):
    ### for scATAC-seq
    def __init__(self, layer, hidden, input_size, Type="Bernoulli", droprate=0.1):
        super().__init__()

        if len(layer) > 1:
            self.decoder = build_multi_layers(layer, dropout_rate=droprate)

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

        elif self.Type == "Gaussian1":
            Final_x = F.softmax(recon_x, dim=1)

        elif self.Type == "Gaussian":
            Final_x = torch.sigmoid(recon_x)

        elif self.Type == "Gaussian2":
            Final_x = F.relu(recon_x)

        else:
            Final_x = recon_x

        return Final_x


class VAE(nn.Module):
    # def __init__( self, layer_e, hidden1, hidden2, layer_l, layer_d, hidden ):
    def __init__(self, layer_e, hidden1, Zdim, layer_d, hidden2, Type='NB', droprate=0.1):

        super().__init__()

        ###  encoder
        self.encoder = Encoder(layer_e, hidden1, Zdim, droprate=droprate)
        self.activation = nn.Softmax(dim=-1)

        ### the decoder
        if Type == 'ZINB':
            self.decoder = DecoderLogNormZINB(layer_d, hidden2, layer_e[0], droprate=droprate)

        elif Type == 'NB':
            self.decoder = DecoderLogNormNB(layer_d, hidden2, layer_e[0], droprate=droprate)

        else:  ## Bernoulli, or Gaussian
            self.decoder = Decoder(layer_d, hidden2, layer_e[0], Type, droprate=droprate)

        ### parameters
        self.Type = Type

    def inference(self, X=None, scale_factor=1.0):
        # encoder
        mean_1, logvar_1, latent_1, hidden = self.encoder.return_all_params(X)

        ### decoder
        if self.Type == 'ZINB':
            output = self.decoder(latent_1, scale_factor)
            norm_x = output["normalized"]
            disper_x = output["disperation"]
            recon_x = output["scale_x"]
            dropout_rate = output["dropoutrate"]

        elif self.Type == 'NB':
            output = self.decoder(latent_1, scale_factor)
            norm_x = output["normalized"]
            disper_x = output["disperation"]
            recon_x = output["scale_x"]
            dropout_rate = None

        else:
            recons_x = self.decoder(latent_1)
            recon_x = recons_x
            norm_x = recons_x
            disper_x = None
            dropout_rate = None

        return dict(norm_x=norm_x, disper_x=disper_x, dropout_rate=dropout_rate, recon_x=recon_x, latent_z1=latent_1,
                    mean_1=mean_1, logvar_1=logvar_1, hidden=hidden)

    def return_loss(self, X=None, X_raw=None, latent_pre=None, mean_pre=None, logvar_pre=None, latent_pre_hidden=None,
                    scale_factor=1.0, cretion_loss=None, attention_loss=None):

        output = self.inference(X, scale_factor)
        recon_x = output["recon_x"]
        disper_x = output["disper_x"]
        dropout_rate = output["dropout_rate"]

        mean_1 = output["mean_1"]
        logvar_1 = output["logvar_1"]
        latent_z1 = output["latent_z1"]

        hidden = output["hidden"]

        if self.Type == 'ZINB':
            loss = log_zinb_positive(X_raw, recon_x, disper_x, dropout_rate)

        elif self.Type == 'NB':
            loss = log_nb_positive(X_raw, recon_x, disper_x)

        elif self.Type == 'Bernoulli':  # here X and X_raw are same
            loss = binary_cross_entropy(recon_x, X_raw)

        else:
            loss = mse_loss(X, recon_x)

        ##calculate KL loss for Gaussian distribution
        mean = torch.zeros_like(mean_1)
        scale = torch.ones_like(logvar_1)
        kl_divergence_z = kl(Normal(mean_1, torch.exp(logvar_1)), Normal(mean, scale)).sum(dim=1)

        atten_loss1 = torch.tensor(0.0)
        if latent_pre is not None and latent_pre_hidden is not None:

            if attention_loss == "KL_div":
                atten_loss1 = cretion_loss(mean_1, logvar_1, mean_pre, logvar_pre)

            else:
                atten_loss1 = cretion_loss(latent_z1, latent_pre)

        return loss, kl_divergence_z, atten_loss1

    def forward(self, X=None, scale_factor=1.0):

        output = self.inference(X, scale_factor)

        return output

    def fit(self, train_loader, test_loader, total_loader, model_pre, args, criterion, cycle, state, first="RNA",
            attention_loss="Eucli"):

        params = filter(lambda p: p.requires_grad, self.parameters())

        if cycle % 2 == 0:
            optimizer = optim.Adam(params, lr=args.lr1, weight_decay=args.weight_decay, eps=args.eps)
        else:
            optimizer = optim.Adam(params, lr=args.lr2, weight_decay=args.weight_decay, eps=args.eps)

        train_loss_list = []
        reco_epoch_test = 0
        test_like_max = 100000
        flag_break = 0

        patience_epoch = 0
        args.anneal_epoch = 10

        model_pre.eval()

        start = time.time()

        for epoch in range(1, args.max_epoch + 1):

            self.train()

            patience_epoch += 1
            kl_weight = min(1, epoch / args.anneal_epoch)

            if cycle % 2 == 0:
                epoch_lr = adjust_learning_rate(args.lr1, optimizer, epoch, args.flr1, 10)
            else:
                epoch_lr = adjust_learning_rate(args.lr2, optimizer, epoch, args.flr2, 10)

            for batch_idx, (X1, X1_raw, size_factor1, X2, X2_raw, size_factor2) in enumerate(train_loader):

                X1, X1_raw, size_factor1 = X1.to(args.device), X1_raw.to(args.device), size_factor1.to(args.device)
                X2, X2_raw, size_factor2 = X2.to(args.device), X2_raw.to(args.device), size_factor2.to(args.device)

                X1, X1_raw, size_factor1 = Variable(X1), Variable(X1_raw), Variable(size_factor1)
                X2, X2_raw, size_factor2 = Variable(X2), Variable(X2_raw), Variable(size_factor2)

                optimizer.zero_grad()

                if first == "RNA":

                    if cycle % 2 == 0:

                        if state == 0:
                            # initialization of scRNA-seq model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X1, X1_raw, None, None, None, None,
                                                                                   size_factor1, criterion,
                                                                                   attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                        else:
                            # transfer representation from scEpigenomics model to scRNA-seq model
                            result_2 = model_pre(X2, size_factor2)
                            latent_z1 = result_2["latent_z1"].to(args.device)
                            hidden_1 = result_2["hidden"].to(args.device)
                            mean_1 = result_2["mean_1"].to(args.device)
                            logvar_1 = result_2["logvar_1"].to(args.device)

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X1, X1_raw, latent_z1, mean_1,
                                                                                   logvar_1, hidden_1, size_factor1,
                                                                                   criterion, attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf2 * (atten_loss1)))

                    else:
                        if state == 0:
                            # initialization of scEpigenomics model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X2, X2_raw, None, None, None, None,
                                                                                   size_factor2, criterion,
                                                                                   attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                        else:
                            # transfer representation form scRNA-seq model to scEpigenomics model
                            result_2 = model_pre(X1, size_factor1)
                            latent_z1 = result_2["latent_z1"].to(args.device)
                            hidden_1 = result_2["hidden"].to(args.device)
                            mean_1 = result_2["mean_1"].to(args.device)
                            logvar_1 = result_2["logvar_1"].to(args.device)

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X2, X2_raw, latent_z1, mean_1,
                                                                                   logvar_1, hidden_1, size_factor2,
                                                                                   criterion, attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf1 * (atten_loss1)))
                else:

                    if cycle % 2 == 0:

                        if state == 0:
                            # initialization of scEpigenomics model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X2, X2_raw, None, None, None, None,
                                                                                   size_factor2, criterion,
                                                                                   attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                        else:
                            # transfer representation from scRNA-seq model to scEpigenomics model
                            result_2 = model_pre(X1, size_factor1)
                            latent_z1 = result_2["latent_z1"].to(args.device)
                            hidden_1 = result_2["hidden"].to(args.device)
                            mean_1 = result_2["mean_1"].to(args.device)
                            logvar_1 = result_2["logvar_1"].to(args.device)

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X2, X2_raw, latent_z1, mean_1,
                                                                                   logvar_1, hidden_1, size_factor2,
                                                                                   criterion, attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf1 * (atten_loss1)))

                    else:
                        if state == 0:
                            # initialization of scRNA-seq model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X1, X1_raw, None, None, None, None,
                                                                                   size_factor1, criterion,
                                                                                   attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                        else:
                            # transfer representation from scEpigenomics model to scRNA-seq model
                            result_2 = model_pre(X2, size_factor2)
                            latent_z1 = result_2["latent_z1"].to(args.device)
                            hidden_1 = result_2["hidden"].to(args.device)
                            mean_1 = result_2["mean_1"].to(args.device)
                            logvar_1 = result_2["logvar_1"].to(args.device)

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss(X1, X1_raw, latent_z1, mean_1,
                                                                                   logvar_1, hidden_1, size_factor1,
                                                                                   criterion, attention_loss)
                            loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf2 * (atten_loss1)))

                loss.backward()
                optimizer.step()

            if epoch % args.epoch_per_test == 0 and epoch > 0:
                self.eval()

                with torch.no_grad():

                    for batch_idx, (X1, X1_raw, size_factor1, X2, X2_raw, size_factor2) in enumerate(test_loader):

                        X1, X1_raw, size_factor1 = X1.to(args.device), X1_raw.to(args.device), size_factor1.to(
                            args.device)
                        X2, X2_raw, size_factor2 = X2.to(args.device), X2_raw.to(args.device), size_factor2.to(
                            args.device)

                        X1, X1_raw, size_factor1 = Variable(X1), Variable(X1_raw), Variable(size_factor1)
                        X2, X2_raw, size_factor2 = Variable(X2), Variable(X2_raw), Variable(size_factor2)

                        if first == "RNA":

                            if cycle % 2 == 0:
                                if state == 0:
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X1, X1_raw, None, None, None, None, size_factor1, criterion, attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                                else:
                                    result_2 = model_pre(X2, size_factor2)
                                    latent_z1 = result_2["latent_z1"].to(args.device)
                                    hidden_1 = result_2["hidden"].to(args.device)
                                    mean_1 = result_2["mean_1"].to(args.device)
                                    logvar_1 = result_2["logvar_1"].to(args.device)

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X1, X1_raw, latent_z1, mean_1, logvar_1, hidden_1, size_factor1, criterion,
                                        attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf2 *
                                                                                                    (atten_loss1)))

                            else:
                                if state == 0:
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X2, X2_raw, None, None, None, None, size_factor2, criterion, attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                                else:
                                    result_2 = model_pre(X1, size_factor1)
                                    latent_z1 = result_2["latent_z1"].to(args.device)
                                    hidden_1 = result_2["hidden"].to(args.device)
                                    mean_1 = result_2["mean_1"].to(args.device)
                                    logvar_1 = result_2["logvar_1"].to(args.device)

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X2, X2_raw, latent_z1, mean_1, logvar_1, hidden_1, size_factor2, criterion,
                                        attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf1 *
                                                                                                    (atten_loss1)))

                        else:
                            if cycle % 2 == 0:

                                if state == 0:
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X2, X2_raw, None, None, None, None, size_factor2, criterion, attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                                else:
                                    result_2 = model_pre(X1, size_factor1)
                                    latent_z1 = result_2["latent_z1"].to(args.device)
                                    hidden_1 = result_2["hidden"].to(args.device)
                                    mean_1 = result_2["mean_1"].to(args.device)
                                    logvar_1 = result_2["logvar_1"].to(args.device)

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X2, X2_raw, latent_z1, mean_1, logvar_1, hidden_1, size_factor2, criterion,
                                        attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf1 *
                                                                                                    (atten_loss1)))

                            else:
                                if state == 0:
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X1, X1_raw, None, None, None, None, size_factor1, criterion, attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z))

                                else:
                                    result_2 = model_pre(X2, size_factor2)
                                    latent_z1 = result_2["latent_z1"].to(args.device)
                                    hidden_1 = result_2["hidden"].to(args.device)
                                    mean_1 = result_2["mean_1"].to(args.device)
                                    logvar_1 = result_2["logvar_1"].to(args.device)

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss(
                                        X1, X1_raw, latent_z1, mean_1, logvar_1, hidden_1, size_factor1, criterion,
                                        attention_loss)
                                    test_loss = torch.mean(loss1 + (kl_weight * kl_divergence_z) + (args.sf2 *
                                                                                                    (atten_loss1)))

                        train_loss_list.append(test_loss.item())

                        print(
                            str(epoch) + "   " + str(test_loss.item()) + "   " + str(torch.mean(loss1).item()) + "   " +
                            str(torch.mean(kl_divergence_z).item()) + "   " + str(torch.mean(atten_loss1).item()))

                        if math.isnan(test_loss.item()):
                            flag_break = 1
                            break

                        if test_like_max > test_loss.item():
                            test_like_max = test_loss.item()
                            reco_epoch_test = epoch
                            patience_epoch = 0
                            best_dict = deepcopy(self.state_dict())

            if flag_break == 1:
                print("containin NA")
                print(epoch)
                break

            if patience_epoch >= 30:
                print("patient with 30")
                print(epoch)
                break

            if len(train_loss_list) >= 2:
                if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-4:
                    print("converged!!!")
                    print(epoch)
                    break

        duration = time.time() - start
        self.load_state_dict(best_dict)

        print('Finish training, total time is: ' + str(duration) + 's')
        self.eval()
        print(self.training)

        print('train likelihood is :  ' + str(test_like_max) + ' epoch: ' + str(reco_epoch_test))


class DCCA(nn.Module):
    """DCCA class.

    Parameters
    ----------
    layer_e_1 : list[int]
        Hidden layer specification for encoder1. List the dimensions of each hidden layer sequentially.
    hidden1_1 : int
        Hidden dimension for encoder1. It should be consistent with the last layer in layer_e_1.
    Zdim_1 : int
        Latent space dimension for VAE1.
    layer_d_1 : list[int]
        Hidden layer specification for decoder1. List the dimensions of each hidden layer sequentially.
    hidden2_1 : int
        Hidden dimension for decoder1. It should be consistent with the last layer in layer_d_1.
    layer_e_2 : int
        Hidden layer specification for encoder2. List the dimensions of each hidden layer sequentially.
    hidden1_2 : int
        Hidden dimension for encoder2. It should be consistent with the last layer in layer_e_1.
    Zdim_2 : int
        Latent space dimension for VAE2.
    layer_d_2 : int
        Hidden layer specification for decoder2. List the dimensions of each hidden layer sequentially.
    hidden2_2 : int
        Hidden dimension for decoder2. It should be consistent with the last layer in layer_d_1.
    args : argparse.Namespace
        A Namespace object that contains arguments of DCCA. For details of parameters in parser args, please refer to link (parser help document).
    ground_truth1 : torch.Tensor
        Extra labels for VAE1.
    Type_1 : str optional
        Loss type for VAE1. Default: 'NB'. By default to be 'NB'.
    Type_2 : str optional
        Loss type for VAE2. Default: 'Bernoulli'. By default to be 'Bernoulli'.
    cycle : int optional
        Number of multiple training cycles. In each cycle iteratively update VAE1 and VAE2. By default to be 1.
    attention_loss : str optional
        Loss type of attention loss. By default to be 'Eucli'.
    droprate : float optional
        Dropout rate for encoder/decoder layers. By default to be 0.1.

    """

    def __init__(self, layer_e_1, hidden1_1, Zdim_1, layer_d_1, hidden2_1, layer_e_2, hidden1_2, Zdim_2, layer_d_2,
                 hidden2_2, args, ground_truth1, Type_1='NB', Type_2='Bernoulli', cycle=1, attention_loss='Eucli',
                 droprate=0.1):

        super().__init__()
        # cycle indicates the mutual learning, 0 for initiation of model1 with scRNA-seq data,
        # and odd for training other models, even for scRNA-seq

        self.model1 = VAE(layer_e=layer_e_1, hidden1=hidden1_1, Zdim=Zdim_1, layer_d=layer_d_1, hidden2=hidden2_1,
                          Type=Type_1, droprate=droprate).to(args.device)
        self.model2 = VAE(layer_e=layer_e_2, hidden1=hidden1_2, Zdim=Zdim_2, layer_d=layer_d_2, hidden2=hidden2_2,
                          Type=Type_2, droprate=droprate).to(args.device)

        if attention_loss == 'NST':
            self.attention = NSTLoss()

        elif attention_loss == 'FT':
            self.attention = FactorTransfer()

        elif attention_loss == 'SL':
            self.attention = Similarity()

        elif attention_loss == 'CC':
            self.attention = Correlation()

        elif attention_loss == 'AT':
            self.attention = Attention()

        elif attention_loss == 'KL_div':
            self.attention = KL_diver()

        elif attention_loss == 'L1':
            self.attention = L1_dis()

        else:
            self.attention = Eucli_dis()

        self.cycle = cycle
        self.args = args
        self.ground_truth1 = ground_truth1.numpy()
        self.attention_loss = attention_loss

    def fit(self, train_loader, test_loader, total_loader, first="RNA"):
        """Fit function for training.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Dataloader for training dataset.
        test_loader : torch.utils.data.DataLoader
            Dataloader for testing dataset.
        total_loader : torch.utils.data.DataLoader
            Dataloader for both training and testing dataset, for extra evaluation purpose.
        first : str
            Type of modality 1.

        Returns
        -------
        None.

        """

        used_cycle = 0

        if self.ground_truth1 is not None:
            self.score(total_loader)

        while used_cycle < (self.cycle + 1):

            if first == "RNA":

                if used_cycle % 2 == 0:

                    self.model2.eval()

                    if used_cycle == 0:

                        self.model1.fit(train_loader, test_loader, total_loader, self.model2, self.args, self.attention,
                                        used_cycle, 0, first, self.attention_loss)

                    else:
                        self.model1.fit(train_loader, test_loader, total_loader, self.model2, self.args, self.attention,
                                        used_cycle, 1, first, self.attention_loss)

                else:
                    self.model1.eval()

                    if used_cycle == 1:

                        self.model2.fit(train_loader, test_loader, total_loader, self.model1, self.args, self.attention,
                                        used_cycle, 0, first, self.attention_loss)

                        if self.ground_truth1 is not None:
                            self.score(total_loader)

                        if self.attention_loss is not None:
                            self.model2.fit(train_loader, test_loader, total_loader, self.model1, self.args,
                                            self.attention, used_cycle, 1, first, self.attention_loss)

                    else:
                        self.model2.fit(train_loader, test_loader, total_loader, self.model1, self.args, self.attention,
                                        used_cycle, 1, first, self.attention_loss)

            else:
                if used_cycle % 2 == 0:

                    self.model1.eval()

                    if used_cycle == 0:

                        self.model2.fit(train_loader, test_loader, total_loader, self.model1, self.args, self.attention,
                                        used_cycle, 0, first, self.attention_loss)

                    else:
                        self.model2.fit(train_loader, test_loader, total_loader, self.model1, self.args, self.attention,
                                        used_cycle, 1, first, self.attention_loss)

                else:
                    self.model2.eval()

                    if used_cycle == 1:

                        self.model1.fit(train_loader, test_loader, total_loader, self.model2, self.args, self.attention,
                                        used_cycle, 0, first, self.attention_loss)

                        if self.ground_truth1 is not None:
                            self.score(total_loader)

                        self.model1.fit(train_loader, test_loader, total_loader, self.model2, self.args, self.attention,
                                        used_cycle, 1, first, self.attention_loss)

                    else:
                        self.model1.fit(train_loader, test_loader, total_loader, self.model2, self.args, self.attention,
                                        used_cycle, 1, first, self.attention_loss)

            used_cycle = used_cycle + 1

    def score(self, dataloader, metric='clustering'):
        """Score function to get score of prediction.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader for testing dataset.

        Returns
        -------
        NMI_score1 : float
            Metric eval score for VAE1.
        ARI_score1 : float
            Metric eval score for VAE1.
        NMI_score2 : float
            Metric eval score for VAE2.
        ARI_score2 : float
            Metric eval score for VAE2.

        """

        if metric == 'clustering':
            self.model1.eval()
            self.model2.eval()

            with torch.no_grad():

                kmeans1 = KMeans(n_clusters=self.args.cluster1, n_init=5, random_state=200)
                kmeans2 = KMeans(n_clusters=self.args.cluster2, n_init=5, random_state=200)

                latent_code_rna = []
                latent_code_atac = []

                for batch_idx, (X1, _, size_factor1, X2, _, size_factor2) in enumerate(dataloader):

                    X1, size_factor1 = X1.to(self.args.device), size_factor1.to(self.args.device)
                    X2, size_factor2 = X2.to(self.args.device), size_factor2.to(self.args.device)

                    X1, size_factor1 = Variable(X1), Variable(size_factor1)
                    X2, size_factor2 = Variable(X2), Variable(size_factor2)

                    result1 = self.model1.inference(X1, size_factor1)
                    result2 = self.model2.inference(X2, size_factor2)

                    latent_code_rna.append(result1["latent_z1"].data.cpu().numpy())
                    latent_code_atac.append(result2["latent_z1"].data.cpu().numpy())

                latent_code_rna = np.concatenate(latent_code_rna)
                latent_code_atac = np.concatenate(latent_code_atac)

                pred_z1 = kmeans1.fit_predict(latent_code_rna)
                NMI_score1 = round(normalized_mutual_info_score(self.ground_truth1, pred_z1, average_method='max'), 3)
                ARI_score1 = round(metrics.adjusted_rand_score(self.ground_truth1, pred_z1), 3)

                pred_z2 = kmeans1.fit_predict(latent_code_atac)
                NMI_score2 = round(normalized_mutual_info_score(self.ground_truth1, pred_z2, average_method='max'), 3)
                ARI_score2 = round(metrics.adjusted_rand_score(self.ground_truth1, pred_z2), 3)

                print('scRNA-ARI: ' + str(ARI_score1) + ' NMI: ' + str(NMI_score1) + ' scEpigenomics-ARI: ' +
                      str(ARI_score2) + ' NMI: ' + str(NMI_score2))
                return NMI_score1, ARI_score1, NMI_score2, ARI_score2
        elif metric == 'openproblems':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _encodeBatch(self, total_loader):
        """Helper function to get latent representation, normalized representation and
        prediction of data.

        Parameters
        ----------
        total_loader : torch.utils.data.DataLoader
            Dataloader for dataset.

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

        # processing large-scale datasets
        latent_z1 = []
        latent_z2 = []
        norm_x1 = []
        recon_x1 = []
        norm_x2 = []
        recon_x2 = []

        for batch_idx, (X1, _, size_factor1, X2, _, size_factor2) in enumerate(total_loader):
            X1, size_factor1 = X1.to(self.args.device), size_factor1.to(self.args.device)
            X2, size_factor2 = X2.to(self.args.device), size_factor2.to(self.args.device)

            X1, size_factor1 = Variable(X1), Variable(size_factor1)
            X2, size_factor2 = Variable(X2), Variable(size_factor2)

            result1 = self.model1(X1, size_factor1)
            result2 = self.model2(X2, size_factor2)

            latent_z1.append(result1["latent_z1"].data.cpu().numpy())
            latent_z2.append(result2["latent_z1"].data.cpu().numpy())

            norm_x1.append(result1["norm_x"].data.cpu().numpy())
            recon_x1.append(result1["recon_x"].data.cpu().numpy())

            norm_x2.append(result2["norm_x"].data.cpu().numpy())
            recon_x2.append(result2["recon_x"].data.cpu().numpy())

        latent_z1 = np.concatenate(latent_z1)
        latent_z2 = np.concatenate(latent_z2)
        norm_x1 = np.concatenate(norm_x1)
        recon_x1 = np.concatenate(recon_x1)
        norm_x2 = np.concatenate(norm_x2)
        recon_x2 = np.concatenate(recon_x2)

        return latent_z1, latent_z2, norm_x1, recon_x1, norm_x2, recon_x2

    def forward(self, total_loader):
        """Forward function for torch.nn.Module. An alias of encode_Batch function.

        Parameters
        ----------
        total_loader : torch.utils.data.DataLoader
            Dataloader for dataset.

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

        latent_z1, latent_z2, norm_x1, recon_x1, norm_x2, recon_x2 = self._encodeBatch(total_loader)

        return latent_z1, latent_z2, norm_x1, recon_x1, norm_x2, recon_x2

    def predict(self, total_loader):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        total_loader : torch.utils.data.DataLoader
            Dataloader for dataset.

        Returns
        -------
        emb1 : numpy.ndarray
            Latent representation of modality 1.
        emb2 : numpy.ndarray
            Latent representation of modality 2.

        """
        self.eval()
        with torch.no_grad():
            emb1, emb2, _, _, _, _ = self.forward(total_loader)

        return emb1, emb2
