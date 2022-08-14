"""Reimplementation of GrapSCI.

Extended from https://github.com/biomed-AI/GraphSCI

Reference
----------
Rao, Jiahua, et al. "Imputing single-cell RNA-seq data by combining graph convolution and autoencoder neural networks."
Iscience 24.5 (2021): 102393.

"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class AEModel(nn.Module):
    """GraphSCI Autoencoder class.

    Parameters
    ----------
    in_feats : int
        dimension of encoder input.
    activation : str optional
        activation function.
    n_hidden1 : int
        dimension of first hidden layer
    n_hidden2 : int
        dimension of second hidden layer

    """

    def __init__(self, in_feats, activation=None, n_hidden1=256, n_hidden2=256):
        super().__init__()
        if activation != None:
            self.act1 = activation
        else:
            self.act1 = nn.Tanh()
        self.act2 = nn.ReLU()
        self.linear0 = nn.Linear(in_feats, in_feats)
        self.linear1 = nn.Linear(in_feats, n_hidden1)
        self.bn1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.bn2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, in_feats)
        self.linear4 = nn.Linear(n_hidden2, in_feats)
        self.linear5 = nn.Linear(n_hidden2, in_feats)

    def forward(self, x, z_adj, size_factors):
        """Forward propagation.

        Parameters
        ----------
        x :
            input gene expression features.
        z_adj:
            adjacency matrix, usually generated from gnn model.
        size_factors:
            cell specific size factors for transforming mean of ZINB to x_exp

        Returns
        -------
        x_exp:
            recreation of input gene expression.
        pi :
            data dropout probability from ZINB.
        disp :
            data dispersion from ZINB.

        """
        x = self.linear0(x)
        x = torch.matmul(z_adj, x)
        x = self.act1(x)
        h = self.linear1(x)
        h = self.bn1(h)
        h = self.act2(h)
        h = self.linear2(h)
        h = self.bn2(h)
        h = self.act2(h)
        pi = torch.sigmoid(self.linear3(h))
        disp = torch.clamp(F.softplus(self.linear4(h)), 1e-4, 1e4)
        mean = torch.clamp(torch.exp(self.linear5(h)), 1e-5, 1e6)
        size_diag = torch.diag(size_factors).to(mean.device)
        x_exp = torch.matmul(size_diag, mean)
        return x_exp, mean, disp, pi


class GNNModel(nn.Module):
    """GraphSCI graphical neural network class.

    Parameters
    ----------
    in_feats : int
        dimension of GNN module input.
    out_feats : int
        dimension of GNN module input.
    n_hidden1 : int optional
        dimension of first hidden layer
    n_hidden2 : int optional
        dimension of second hidden layer
    activation : str optional
        activation function.
    dropout : float optional
        probability of weight dropout for training.

    """

    def __init__(self, in_feats, out_feats, n_hidden1=256, n_hidden2=256, activation=None, dropout=0.0):
        self.in_feats = in_feats
        self.out_feats = out_feats
        super().__init__()
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.linear1 = nn.Linear(in_feats, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_feats)
        self.linear4 = nn.Linear(n_hidden2, out_feats)
        if activation != None:
            self.act = activation
        else:
            self.act = nn.ReLU()

    def forward(self, x, adj):
        """Forward propagation.
        Parameters
        ----------
        x :
            input gene expression features.
        adj:
            gene graph adjacency matrix
        Returns
        -------
        z_adj :
            recreation of adjacency matrix
        z_adj_std :
            standard deviation parameter of normal distribution
        z_adj_mean :
            data dispersion from ZINB.
        """
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.matmul(adj, x)
        h1 = self.act(x)
        h1 = self.dropout(h1)
        h1 = self.linear2(h1)
        h1 = torch.matmul(adj, h1)
        h2 = self.act(h1)
        z_adj_mean = self.dropout(h2)
        z_adj_mean = self.linear3(z_adj_mean)
        z_adj_mean = torch.matmul(adj, z_adj_mean)
        z_adj_mean = torch.matmul(z_adj_mean, z_adj_mean.t())
        z_adj_std = self.dropout(h2)
        z_adj_std = self.linear4(z_adj_std)
        z_adj_std = torch.matmul(adj, z_adj_std)
        z_adj_std = torch.matmul(z_adj_std, z_adj_std.t())
        z_adj = z_adj_mean + torch.normal(z_adj_std, torch.ones(x.shape[0], x.shape[0]).to(z_adj_std.device))
        return z_adj, z_adj_std, z_adj_mean


class GraphSCI:
    """GraphSCI model, combination AE and GNN.

    Parameters
    ----------
    num_cells : int
        number of cells in expression data
    num_genes : int
        number of genes in expression data
    train_dataset : str
        name of training dataset
    n_epochs : int optional
        number of training epochs
    lr : float optional
        learning rate
    weight_decay : float optional
        weight decay rate
    dropout : float optional
        probability of weight dropout for training
    gpu: int optional
        index of computing device, -1 for cpu.

    """

    def __init__(self, num_cells, num_genes, train_dataset, n_epochs=100, lr=1e-3, weight_decay=1e-5, dropout=0.1,
                 gpu=-1):
        # self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.prj_path = Path(
            __file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve().parent.resolve()
        self.train_dataset = train_dataset
        self.save_path = self.prj_path / 'example' / 'single_modality' / 'imputation' / 'pretrained' / train_dataset / 'models'
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.gnnmodel = GNNModel(in_feats=self.num_cells, out_feats=self.num_cells,
                                 dropout=self.dropout).to(self.device)
        self.aemodel = AEModel(in_feats=self.num_cells, ).to(self.device)
        self.model_params = list(self.aemodel.parameters()) + list(self.gnnmodel.parameters())
        self.optimizer = torch.optim.Adam(self.model_params, lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, train_data, train_data_raw, adj_train, train_size_factors, adj_norm_train, le=1, la=1, ke=1, ka=1):
        min_train_loss, _train_loss, _epoch = 1000, 1000, 1
        """ Data fitting function
        Parameters
        ----------
        train_data :
            input training features
        train_data_raw :
            input raw training features
        adj_train :
            training adjacency matrix of gene graph
        train_size_factors :
            train size factors for cells
        adj_norm_train :
            normalized training adjacency matrix of gene graph
        le : float optioanl
            parameter of expression loss
        la : float optioanl
            parameter of adjacency loss
        ke : float optioanl
            parameter of KL divergence of expression
        ka : float optioanl
            parameter of KL divergence of adjacency

        Returns
        -------
        None
        """

        for epoch in range(self.n_epochs):
            total_loss = self.train(train_data, train_data_raw, adj_train, train_size_factors, adj_norm_train, le, la,
                                    ke, ka)
            train_loss, z_adj_train, z_exp_train = self.evaluate(train_data, train_data_raw, adj_norm_train, adj_train,
                                                                 train_size_factors, le, la, ke, ka)
            if min_train_loss <= train_loss:
                min_train_loss = train_loss
                _train_loss = train_loss
                _epoch = epoch
                self.save_model()
            print(f"[Epoch%d], train_loss %.6f, adj_loss %.6f, express_loss %.6f, kl loss %.6f" \
                  % (epoch, self.loss, self.loss_adj, self.loss_exp, abs(self.kl)))

    def train(self, train_data, train_data_raw, adj_orig, size_factors, adj_norm, le=1, la=1, ke=1, ka=1):
        """ Train function, gets loss and performs optimization step
        Parameters
        ----------
        train_data :
            input training features
        train_data_raw :
            input raw training features
        adj_orig :
            training adjacency matrix of gene graph
        size_factors :
            train size factors for cells
        adj_norm :
            normalized training adjacency matrix of gene graph
        le : float optioanl
            parameter of expression loss
        la : float optioanl
            parameter of adjacency loss
        ke : float optioanl
            parameter of KL divergence of expression
        ka : float optioanl
            parameter of KL divergence of adjacency


        Returns
        -------
        total_loss : float
            loss value of training loop
        """
        self.gnnmodel.train()
        self.aemodel.train()
        self.optimizer.zero_grad()
        z_adj, z_adj_std, z_adj_mean = self.gnnmodel.forward(train_data, adj_norm)
        z_exp, mean, disp, pi = self.aemodel.forward(train_data, z_adj, size_factors)
        loss_adj, loss_exp, log_lik, kl, loss = self.get_loss(train_data_raw, adj_orig, z_adj, z_adj_std, z_adj_mean,
                                                              z_exp, mean, disp, pi, size_factors, le, la, ke, ka)
        self.loss_adj = loss_adj.item()
        self.loss_exp = loss_exp.item()
        self.log_lik = log_lik.item()
        self.kl = kl.item()
        self.loss = loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, 1e04)
        self.optimizer.step()

        total_loss = self.loss
        del loss_adj, loss_exp, log_lik, kl, loss

        return total_loss

    def evaluate(self, features, features_raw, adj_norm, adj_orig, size_factors, le=1, la=1, ke=1, ka=1):
        """ Evaluate function, returns loss and reconstructions of expression and adjacency
        Parameters
        ----------
        features :
            input features
        features_raw :
            input raw features
        adj_norm :
            normalized adjacency matrix of gene graph
        adj_orig :
            training adjacency matrix of gene graph
        size_factors :
            cell size factors for reconstruction
        le : float optioanl
            parameter of expression loss
        la : float optioanl
            parameter of adjacency loss
        ke : float optioanl
            parameter of KL divergence of expression
        ka : float optioanl
            parameter of KL divergence of adjacency


        Returns
        -------
        """
        self.aemodel.eval()
        self.gnnmodel.eval()
        with torch.no_grad():
            z_adj, z_adj_std, z_adj_mean = self.gnnmodel.forward(features, adj_norm)
            z_exp, mean, disp, pi = self.aemodel.forward(features, z_adj, size_factors)
            loss_adj, loss_exp, log_lik, kl, loss = self.get_loss(features_raw, adj_orig, z_adj, z_adj_std, z_adj_mean,
                                                                  z_exp, mean, disp, pi, size_factors, le, la, ke, ka)
        return loss, z_adj, z_exp

    def save_model(self):
        """Save model function, saves both AE and GNN."""
        state = {
            'aemodel': self.aemodel.state_dict(),
            'gnnmodel': self.gnnmodel.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(state, self.save_path / f"{self.train_dataset}.pt")

    def predict(self, data, data_raw, adj_norm, adj_orig, size_factors):
        """ Predict function
        Parameters
        ----------
        data :
            input true expression data
        data_raw :
            raw input true expression data
        adj_norm :
            normalized adjacency matrix of gene graph
        adj_orig :
            adjacency matrix of gene graph
        size_factors :
            cell size factors for reconstruction

        Returns
        -------
        z_exp :
            reconstructed expression data
        """
        _, _, z_exp = self.evaluate(data, data_raw, adj_norm, adj_orig, size_factors)
        return z_exp

    def get_loss(self, batch, adj_orig, z_adj, z_adj_std, z_adj_mean, z_exp, mean, disp, pi, sf, le=1, la=1, ke=1,
                 ka=1):
        """ Loss function for GraphSCI
        Parameters
        ----------
        batch :
            batch features
        z_adj :
            reconstructed adjacency matrix
        z_adj_std :
            standard deviation of distribution of z_adj
        z_adj_mean :
            mean of distributino of z_adj
        z_exp :
            recontruction of expression values
        mean :
            dropout parameter of ZINB dist of z_exp
        disp :
            dropout parameter of ZINB dist of z_exp
        pi :
            dispersion parameter of ZINB dist of z_exp
        sf :
            cell size factors
        le : float optioanl
            parameter of expression loss
        la : float optioanl
            parameter of adjacency loss
        ke : float optioanl
            parameter of KL divergence of expression
        ka : float optioanl
            parameter of KL divergence of adjacency

        Returns
        -------
        loss_adj : float
            loss of adjacency reconstruction
        loss_exp : float
            loss of expression reconstruction
        log_lik : float
            log likelihood loss value
        kl : float
            kullback leibler loss
        loss : float
            log_lik - kl
        """

        pos_weight = (adj_orig.shape[0]**2 - adj_orig.sum(axis=1)) / (adj_orig.sum(axis=1))
        norm_adj = (adj_orig.shape[0]**2) / (adj_orig.shape[0]**2 - adj_orig.sum() * 2)
        loss_adj = la * torch.mean(F.cross_entropy(z_adj, adj_orig)) / adj_orig.shape[0]

        eps = 1e-7
        t1 = torch.lgamma(disp + eps) + torch.lgamma(batch + 1) - torch.lgamma(batch + disp + eps)
        t2 = (disp + batch) * torch.log(1.0 + (mean / (disp + eps))) + (batch *
                                                                        (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_loss = t1 + t2
        nb_loss = torch.where(torch.isnan(nb_loss),
                              torch.zeros([nb_loss.shape[0], nb_loss.shape[1]]).to(self.device) + np.inf, nb_loss)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1 - pi) * zero_nb) + eps)
        loss_exp = torch.where(torch.lt(batch, 1e-8), zero_case, nb_loss)
        loss_exp = le * torch.mean(loss_exp)
        log_lik = loss_exp + loss_adj

        kl_adj = (self.weight_decay * 0.5 / batch.shape[1]**2) * torch.mean(
            torch.sum(1 + 2 * torch.log(abs(z_adj_std)) - torch.square(z_adj_mean) - torch.square(z_adj_std), 1))
        # kl_exp = (0.5 / batch.shape[0]) * torch.mean(F.mse_loss(z_exp, batch))
        kl_exp = 0.5 * torch.mean(F.mse_loss(z_exp, batch))
        kl = ka * kl_adj - ke * kl_exp
        loss = log_lik - kl
        return loss_adj, loss_exp, log_lik, kl, loss

    def load_model(self):
        """Load function."""
        model_path = self.prj_path / 'pretrained' / self.train_dataset / 'models' / f'{self.train_dataset}.pt'
        state = torch.load(model_path, map_location=self.device)
        self.aemodel.load_state_dict(state['aemodel'])
        self.gnnmodel.load_state_dict(state['gnnmodel'])

    def score(self, true_expr, imputed_expr, test_idx, metric="MSE"):
        """ Scoring function of model
        Parameters
        ----------
        true_expr :
            True underlying expression values
        imputed_expr :
            Imputed expression values
        test_idx :
            index of testing genes
        metric : str optional
            Choice of scoring metric - 'MSE' or 'ARI'

        Returns
        -------
        mse_error : float
            mean square error
        """
        allowd_metrics = {"MSE", "PCC"}
        true_target = true_expr[test_idx, ]
        imputed_target = imputed_expr[test_idx, ]
        if metric not in allowd_metrics:
            raise ValueError("scoring metric %r." % allowd_metrics)

        if (metric == 'MSE'):
            mse_cells = pd.DataFrame(((true_target.cpu() - imputed_target.cpu())**2).mean(axis=0)).dropna()
            mse_genes = pd.DataFrame(((true_target.cpu() - imputed_target.cpu())**2).mean(axis=1)).dropna()
            return mse_cells, mse_genes
        elif (metric == 'PCC'):
            # cor_cells = np.corrcoef(true_target.cpu(), imputed_target.cpu())
            cor_cells = np.corrcoef(true_expr.cpu(), imputed_expr.cpu())
            return cor_cells
