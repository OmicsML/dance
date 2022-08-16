"""A PyTorch reimplementation of the DSTG cell-type deconvolution method.

Adapted from https://github.com/Su-informatics-lab/DSTG

Reference
---------
Song, and Su. "DSTG: deconvoluting spatial transcriptomics data through graph-based artificial intelligence."
Briefings in Bioinformatics (2021)

"""
import math
import time

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from dance.transforms.graph_construct import stAdjConstruct
from dance.transforms.preprocess import preprocess_adj, pseudo_spatial_process, split


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, support, bias=False):
        """GraphConvolution.

        Parameters
        ----------
        in_features : int
            input dimension.
        out_features : int
            output dimension.
        support :
            support for graph convolution.
        bias : boolean optional
            include bias term, default False.

        Returns
        -------
        None.

        """
        super().__init__()
        self.support = support
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        #glorot
        init_range = np.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-init_range, init_range)
        # similar to kaiming normal/uniform
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """forward function.

        Parameters
        ----------
        input :
            node features.
        adj :
            adjacency matrix.

        Returns
        -------
        output : float
            output of graph convolution layer.

        """
        #convolution
        if input.is_sparse:
            #sparse input features
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)

        #adj should always be sparse
        #### add a ReLU or other activation!!
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """dropout + GC + activation."""

    def __init__(self, nfeat, nhid1, nout, bias=False, dropout=0., act=F.relu):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1, bias)
        self.gc2 = GraphConvolution(nhid1, nout, bias)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj):
        """forward function.

        Parameters
        ----------
        x :
            node features.
        adj :
            adjacency matrix.

        Returns
        -------
        output : float
            output of graph convolution network.

        """
        #self.nnz = x._nnz() if sparse_inputs else None

        #dropout + convolution
        x = dropout_layer(x, self.dropout)
        x = self.gc1(x, adj)
        x = self.act(x)
        x = dropout_layer(x, self.dropout)
        x = self.gc2(x, adj)

        return x


class DSTGLearner:
    """DSTGLearner.

    Parameters
    ----------
    nfeat : int
        input dimension.
    nhid1 : int
        number of units in the hidden layer (graph convolution).
    nout : int
        output dimension.
    bias : boolean optional
        include bias term, default False.
    dropout : float optional
        dropout rate, default 0.
    act : optional
        activation function, default torch functional relu.

    Returns
    -------
    None.

    """

    def __init__(self, sc_count, sc_annot, scRNA, mix_count, clust_vr, mix_annot=None, n_hvg=2000, N_p=1000, k_filter=0,
                 nhid=32, bias=False, dropout=0., device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device

        #set adata objects for sc ref and cell mixtures
        sc_adata = sc.AnnData(sc_count)
        sc_adata.obs = sc_annot
        mix_adata = sc.AnnData(mix_count)
        #mix_adata.obs = true_p

        #pre-process: get variable genes --> normalize --> log1p --> standardize --> out
        #set scRNA to false if already using pseudo spot data with real spot data
        #set to true if the reference data is scRNA (to be used for generating pseudo spots)
        mix_counts, mix_labels, hvgs = pseudo_spatial_process([sc_adata, mix_adata], [sc_annot, mix_annot], clust_vr,
                                                              scRNA, n_hvg, N_p)
        mix_labels = [lab.drop(['cell_count', 'total_umi_count', 'n_counts'], axis=1) for lab in mix_labels]

        # create train/val/test split
        adj_data, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label = split(
            mix_counts, mix_labels, pre_process=1, split_val=.8)

        self.labels_binary_train = torch.FloatTensor(labels_binary_train).to(device)
        self.features = torch.sparse.FloatTensor(
            torch.LongTensor([features[0][:, 0].tolist(), features[0][:, 1].tolist()]),
            torch.FloatTensor(features[1])).to(device)
        self.train_mask = torch.FloatTensor(train_mask).to(device)

        #construct adjacency matrix
        adj = stAdjConstruct(mix_counts, mix_labels, adj_data, k_filter=k_filter)
        #preprocess adjacency matrix
        adj = preprocess_adj(adj)

        self.adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                                            torch.FloatTensor(adj.data.astype(np.int32))).to(device)

        nfeat = self.features.size()[1]
        nout = self.labels_binary_train.size()[1]
        #initialize GCN module
        self.model = GCN(nfeat, nhid, nout, bias, dropout).to(device)

    def fit(self, lr=0.005, max_epochs=50, weight_decay=0):
        """fit function for model training.

        Parameters
        ----------
        lr : float optional
            learning rate.
        max_epochs : int optional
            maximum number of epochs to train.
        weight_decay : float optional
            weight decay parameter for optimization (Adam).

        Returns
        -------
        None.

        """
        X = self.features  # node features
        adj = self.adj  # ajacency matrix
        labels = self.labels_binary_train  # labels of pseudo spots (and real spots if provided)
        labels_mask = self.train_mask  # mask to indicate which samples to use for training

        #device = self.device
        model = self.model
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(max_epochs):
            t = time.time()

            y_hat = model(X, adj)
            loss = masked_softmax_cross_entropy(y_hat, labels, labels_mask)

            if (epoch + 1) % 5 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss), "time=",
                      "{:.5f}".format(time.time() - t))
            #    _ = model(X, adj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self):
        """prediction function.
        Parameters
        ----------

        Returns
        -------

        pred : torch tensor
            predictions of cell-type proportions.

        """
        self.model.eval()
        X = self.features
        adj = self.adj
        fX = self.model(X, adj)
        pred = F.softmax(fX, dim=1)
        return pred

    def score(self, pred, true_prop, score_metric='ce'):
        """Model performance score.

        Parameters
        ----------
        pred :
            predicted cell-type proportions.
        true_prop :
            true cell-type proportions.
        score_metric :
            metric used to assess prediction performance.

        Returns
        -------
        loss : float
            loss between predicted and true labels.

        """
        true_prop = true_prop.to(self.device)
        self.model.eval()
        if score_metric == 'ce':
            loss = F.cross_entropy(pred, true_prop)
        elif score_metric == 'mse':
            loss = ((pred / torch.sum(pred, 1, keepdims=True) -
                     true_prop / torch.sum(true_prop, 1, keepdims=True))**2).mean()
        return loss.detach().item()


def dropout_layer(x, dropout):
    """dropout_layer.
    Parameters
    ----------
    x:
        input to dropout layer.
    dropout : float
        dropout rate (between 0 and 1).

    Returns
    -------
    out : torch tensor
        dropout output.
    """
    if x.is_sparse:
        #sparse input features
        out = sparse_dropout(x, dropout)
        return out
    else:
        out = F.dropout(x, dropout)
        return out


def sparse_dropout(x, dropout):
    """sparse_dropout.
    Parameters
    ----------
    x:
        input to dropout layer.
    dropout : float
        dropout rate (between 0 and 1).

    Returns
    -------
    out * (1. / (1-dropout)) : torch tensor
        dropout output.
    """

    noise_shape = x._nnz()
    random_tensor = (1 - dropout) + torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    return out * (1. / (1 - dropout))


"""Softmax cross-entropy loss with masking."""


def masked_softmax_cross_entropy(preds, labels, mask):
    """masked_softmax_cross_entropy.
    Parameters
    ----------
    preds:
        cell-type proportion predictions from dstg model.
    labels :
        true cell-type proportion labels.
    mask :
        mask to indicate which samples to use in computing loss.

    Returns
    -------
    loss : float
        cross entropy loss between true and predicted cell-type proportions (mean reduced).
    """
    if (mask is None):
        loss = F.cross_entropy(preds, labels, reduction='mean')
        return loss
    else:
        loss = F.cross_entropy(preds, labels, reduction='none')
        mask = mask.type(torch.float32)
        mask /= torch.mean(mask)
        loss *= mask
        loss = torch.mean(loss)
        return loss
