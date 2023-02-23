"""Reimplementation of GrapSCI.

Extended from https://github.com/biomed-AI/GraphSCI

Reference
----------
Rao, Jiahua, et al. "Imputing single-cell RNA-seq data by combining graph convolution and autoencoder neural networks."
Iscience 24.5 (2021): 102393.

"""

import time
from pathlib import Path

import scanpy as sc
import numpy as np
import pandas as pd
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dance.modules.base import BaseRegressionMethod
from dance.transforms import AnnDataTransform, Compose, SaveRaw, SetConfig, CellwiseMaskData
from dance.transforms.graph import FeatureFeatureGraph
from dance.typing import Any, List, LogLevel, Optional, Tuple


def buildNetwork(layers, dropout=0., activation=nn.ReLU()):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Dropout(dropout))
        net.append(nn.Linear(layers[i - 1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
    net = nn.Sequential(*net)
    return net


class MultiplyLayer(nn.Module):

    def __init__(self, num_nodes, dropout=0., act=nn.ReLU(), bias=True):
        super(MultiplyLayer, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.act = act
        self.fc_layer = nn.Linear(num_nodes, num_nodes, bias=False)
        self.dp = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_nodes))
    
    def forward(self, X, adj):
        z = self.fc_layer(adj)
        z = torch.matmul(self.dp(X), z)
        if self.bias:
            z = z + self.bias
        return self.act(z)


class AEModel(nn.Module):

    def __init__(self, in_feats, dropout=0., n_hidden1=256, n_hidden2=256):
        super().__init__()
        self.mul_layer = MultiplyLayer(in_feats, dropout)
        self.enc = buildNetwork([in_feats, n_hidden1, n_hidden2], dropout)
        self.dec_pi = buildNetwork([n_hidden2, in_feats], dropout, nn.Sigmoid())
        self.dec_disp = buildNetwork([n_hidden2, in_feats], dropout, 
                                     lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4))
        self.dec_mean = buildNetwork([n_hidden2, in_feats], dropout,
                                     lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6))

    def forward(self, X, adj, size_factors):
        h = self.mul_layer(X, adj)
        h = self.enc(h)
        pi = self.dec_pi(h)
        disp = self.dec_disp(h)
        mean = self.dec_mean(h)
        x_exp = mean * torch.reshape(size_factors, (-1,1))
        return x_exp, mean, disp, pi


class GNNModel(nn.Module):

    def __init__(self, in_feats, out_feats, dropout=0., n_hidden1=256, n_hidden2=256):
        super().__init__()
        self.dp = nn.Dropout(dropout)
        self.conv1 = dglnn.GraphConv(in_feats, n_hidden1, activation=nn.Tanh())
        self.conv2 = dglnn.GraphConv(n_hidden1, n_hidden2, activation=nn.ReLU())
        self.dec_mean = dglnn.GraphConv(n_hidden2, out_feats)
        self.dec_log_std = dglnn.GraphConv(n_hidden2, out_feats)

    def forward(self, g):
        h = self.conv1(g, self.dp(g.ndata["feat"]))
        h = self.conv2(g, self.dp(h))
        z_adj_mean = self.dec_mean(g, self.dp(h))
        z_adj_log_std = self.dec_mean(g, self.dp(h))
        z_adj = torch.normal(z_adj_mean, torch.exp(z_adj_log_std))
        return z_adj, z_adj_log_std, z_adj_mean


class GraphSCI(nn.Module, BaseRegressionMethod):
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

    def __init__(self, num_cells, num_genes, train_dataset, dropout=0.1, gpu=-1):
        # self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.train_dataset = train_dataset
        self.prj_path = Path(__file__)
        self.save_path = self.prj_path / "graphsci" / train_dataset / "models"
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
        self.gnnmodel = GNNModel(in_feats=num_genes, out_feats=num_genes,dropout=dropout)
        self.aemodel = AEModel(in_feats=num_cells, dropout=dropout)
        self.model_params = list(self.aemodel.parameters()) + list(self.gnnmodel.parameters())
        self.to(self.device)
    
    @staticmethod
    def preprocessing_pipeline(threshold: float = 0.3, normalize_edges: bool = True, distr: str = "exp",
                               mask_rate: float = 0.1, seed: int = 1, log_level: LogLevel = "INFO"):
        return Compose(
            AnnDataTransform(sc.pp.filter_genes, min_counts=1),
            AnnDataTransform(sc.pp.filter_cells, min_counts=1),
            SaveRaw(),
            AnnDataTransform(sc.pp.normalize_total),
            AnnDataTransform(sc.pp.log1p),
            FeatureFeatureGraph(threshold=threshold, normalize_edges=normalize_edges),
            CellwiseMaskData(distr=distr, mask_rate=mask_rate, seed=seed),
            SetConfig({
                "feature_channel": ["X", "X_raw", "n_counts", "FeatureFeatureGraph"],
                "feature_channel_type": ["layers", "layers", "obs", "uns"],
                "label_channel": ["label", "label_raw"], 
                "label_channel_type": ["layers", "layers"], 
            }),
            log_level=log_level,
        )

    def fit(self, train_data, train_data_raw, label, label_raw, n_counts, graph, le=1, la=1, ke=1, ka=1, n_epochs=100, lr=1e-3, weight_decay=1e-5):
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
        # Get weighted adjacency matrix
        u, v = graph.edges()
        self.adj = torch.zeros((graph.num_nodes(), graph.num_nodes()))
        self.adj_norm = torch.zeros((graph.num_nodes(), graph.num_nodes()))
        self.adj[u, v] = torch.ones(graph.num_edges()).float()
        self.adj_norm[u, v] = graph.edata['weight']

        self.size_factors = torch.tensor(n_counts / np.median(n_counts))
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.model_params, lr=lr, weight_decay=weight_decay)
        for epoch in range(n_epochs):
            total_loss = self.train(train_data, label, label_raw, graph, le, la,
                                    ke, ka)
            # if min_train_loss >= train_loss:
            #     min_train_loss = train_loss
            #     self.save_model()
            print(f"[Epoch%d], train_loss %.6f, adj_loss %.6f, express_loss %.6f, kl loss %.6f" \
                  % (epoch, self.loss, self.loss_adj, self.loss_exp, abs(self.kl)))

    def train(self, train_data, train_data_raw, graph, le=1, la=1, ke=1, ka=1):
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
        z_adj, z_adj_std, z_adj_mean = self.gnnmodel.forward(graph)
        z_exp, mean, disp, pi = self.aemodel.forward(train_data, z_adj, size_factors)
        loss_adj, loss_exp, log_lik, kl, loss = self.get_loss(train_data_raw, adj_orig, z_adj, z_adj_std, z_adj_mean,
                                                              z_exp, mean, disp, pi, size_factors, le, la, ke, ka)
        self.loss_adj = loss_adj.item()
        self.loss_exp = loss_exp.item()
        self.log_lik = log_lik.item()
        self.kl = kl.item()
        self.loss = loss.item()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model_params, 1e04)
        self.optimizer.step()

        total_loss = self.loss
        del loss_adj, loss_exp, log_lik, kl, loss

        return total_loss

    def evaluate(self, features, features_raw, graph, adj_orig, size_factors, le=1, la=1, ke=1, ka=1):
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
            z_adj, z_adj_std, z_adj_mean = self.gnnmodel.forward(graph)
            z_exp, mean, disp, pi = self.aemodel.forward(features, z_adj, size_factors)
            _, _, _, _, loss = self.get_loss(features_raw, adj_orig, z_adj, z_adj_std, z_adj_mean,
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

    def predict(self, data, data_raw, graph, adj_orig, size_factors):
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
        _, _, z_exp = self.evaluate(data, data_raw, graph, adj_orig, size_factors)
        return z_exp

    def get_loss(self, batch, adj_orig, z_adj, z_adj_log_std, z_adj_mean, z_exp, mean, 
                 disp, pi, sf, le=1, la=1, ke=1, ka=1):
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
        loss_adj = la * torch.mean(F.cross_entropy(z_adj, adj_orig, pos_weight)) / adj_orig.shape[0]

        eps = 1e-10
        disp = torch.clamp(disp, max=1e6)
        t1 = torch.lgamma(disp + eps) + torch.lgamma(batch + 1) - torch.lgamma(batch + disp + eps)
        t2 = (disp + batch) * torch.log(1.0 + (z_exp / (disp + eps))) + (batch *
                                                                        (torch.log(disp + eps) - torch.log(z_exp + eps)))
        nb_loss = t1 + t2
        nb_loss = torch.where(torch.isnan(nb_loss),
                              torch.zeros([nb_loss.shape[0], nb_loss.shape[1]]).to(self.device) + np.inf, nb_loss)
        zero_nb = torch.pow(disp / (disp + z_exp + eps), disp)
        zero_case = -torch.log(pi + ((1 - pi) * zero_nb) + eps)
        loss_exp = torch.where(torch.lt(batch, 1e-8), zero_case, nb_loss)
        loss_exp = le * torch.mean(loss_exp)
        log_lik = loss_exp + loss_adj

        kl_adj = ( 0.5 / batch.shape[0]) * torch.mean(
            torch.sum(1 + 2 * z_adj_log_std - torch.square(z_adj_mean) - torch.square(torch.exp(z_adj_log_std)), 1))
        kl_exp = self.weight_decay * 0.5 / batch.shape[1] * torch.mean(F.mse_loss(z_exp, batch))
        kl = ka * kl_adj - ke * kl_exp
        loss = log_lik - kl
        return loss_adj, loss_exp, log_lik, kl, loss

    def load_model(self):
        """Load function."""
        model_path = self.save_path / f"{self.train_dataset}.pt"
        state = torch.load(model_path, map_location=self.device)
        self.aemodel.load_state_dict(state['aemodel'])
        self.gnnmodel.load_state_dict(state['gnnmodel'])

    def score(self, true_expr, imputed_expr, metric="MSE"):
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
        if metric not in allowd_metrics:
            raise ValueError("scoring metric %r." % allowd_metrics)

        if (metric == 'MSE'):
            mse_cells = pd.DataFrame(((true_expr.cpu() - imputed_expr.cpu())**2).mean(axis=0)).dropna()
            mse_genes = pd.DataFrame(((true_expr.cpu() - imputed_expr.cpu())**2).mean(axis=1)).dropna()
            return mse_cells, mse_genes
        elif (metric == 'PCC'):
            cor_cells = np.corrcoef(true_expr.cpu(), imputed_expr.cpu())
            return cor_cells
