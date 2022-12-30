"""A PyTorch reimplementation of the DSTG cell-type deconvolution method.

Adapted from https://github.com/Su-informatics-lab/DSTG

Reference
---------
Song, and Su. "DSTG: deconvoluting spatial transcriptomics data through graph-based artificial intelligence."
Briefings in Bioinformatics (2021)

"""
import time

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from dance.transforms.graph.dstg_graph import compute_dstg_adj
from dance.transforms.preprocess import pseudo_spatial_process
from dance.utils.matrix import normalize


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, support, bias=False):
        """GraphConvolution.

        Parameters
        ----------
        in_features : int
            Input dimension.
        out_features : int
            Output dimension.
        support :
            Support for graph convolution.
        bias : boolean optional
            Include bias term, default False.

        """
        super().__init__()
        self.support = support
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # glorot
        init_range = np.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-init_range, init_range)
        # similar to kaiming normal/uniform
        stdv = 1. / np.sqrt(self.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """Forward function.

        Parameters
        ----------
        x
            Node features.
        adj
            Adjacency matrix.

        Returns
        -------
        output : float
            Output of graph convolution layer.

        """
        # Convolution
        if x.is_sparse:  # Sparse x features
            support = torch.spmm(x, self.weight)
        else:
            support = torch.mm(x, self.weight)

        # Adj should always be sparse
        # Add a ReLU or other activation!!
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"


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
        x
            Node features.
        adj
            Adjacency matrix.

        Returns
        -------
        output : float
            Output of graph convolution network.

        """
        # Dropout + convolution
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
        Input dimension.
    nhid1 : int
        Number of units in the hidden layer (graph convolution).
    nout : int
        Output dimension.
    bias : boolean optional
        Include bias term, default False.
    dropout : float optional
        Dropout rate, default 0.
    act : optional
        Activation function, default torch functional relu.

    """

    def __init__(self, sc_count, sc_annot, scRNA, mix_count, clust_vr, mix_annot=None, n_hvg=2000, N_p=1000, k_filter=0,
                 nhid=32, bias=False, dropout=0., device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device

        # Set adata objects for sc ref and cell mixtures
        sc_adata = sc.AnnData(sc_count, obs=sc_annot, dtype=np.float32)
        mix_adata = sc.AnnData(mix_count, dtype=np.float32)

        # pre-process: get variable genes --> normalize --> log1p --> standardize --> out
        # set scRNA to false if already using pseudo spot data with real spot data
        # set to true if the reference data is scRNA (to be used for generating pseudo spots)
        mix_counts, mix_labels, hvgs = pseudo_spatial_process([sc_adata, mix_adata], [sc_annot, mix_annot], clust_vr,
                                                              scRNA, n_hvg, N_p)
        mix_labels = [lab.drop(["cell_count", "total_umi_count", "n_counts"], axis=1) for lab in mix_labels]

        features = np.vstack((mix_counts[0].X, mix_counts[1].X)).astype(np.float32)
        normalized_features = normalize(features, axis=1, mode="normalize")
        labels = np.vstack(mix_labels).astype(np.float32)
        train_mask = np.zeros(len(labels), dtype=np.bool)
        train_mask[:len(mix_counts[0])] = True

        self.labels_binary_train = torch.from_numpy(labels).to(device)
        self.features = torch.from_numpy(normalized_features).to(device)
        self.train_mask = torch.from_numpy(train_mask).to(device)

        # Construct and process adjacency matrix
        pseudo_mix_counts, real_mix_counts = mix_counts
        adj = compute_dstg_adj(pseudo_mix_counts, real_mix_counts, k_filter=k_filter)
        self.adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                                            torch.FloatTensor(adj.data.astype(np.int32))).to(device)

        # Initialize GCN module
        nfeat = self.features.size()[1]
        nout = self.labels_binary_train.size()[1]
        self.model = GCN(nfeat, nhid, nout, bias, dropout).to(device)

    def fit(self, lr=0.005, max_epochs=50, weight_decay=0):
        """Fit function for model training.

        Parameters
        ----------
        lr : float optional
            Learning rate.
        max_epochs : int optional
            Maximum number of epochs to train.
        weight_decay : float optional
            Weight decay parameter for optimization (Adam).

        """
        X = self.features  # node features
        adj = self.adj  # ajacency matrix
        labels = self.labels_binary_train  # labels of pseudo spots (and real spots if provided)
        labels_mask = self.train_mask  # mask to indicate which samples to use for training

        model = self.model
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(max_epochs):
            t = time.time()

            y_hat = model(X, adj)
            loss = masked_softmax_cross_entropy(y_hat, labels, labels_mask)

            if (epoch + 1) % 5 == 0:
                print("Epoch:", "%04d" % (epoch + 1), "train_loss=", "{:.5f}".format(loss), "time=",
                      "{:.5f}".format(time.time() - t))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self):
        """Prediction function.

        Returns
        -------
        pred : torch tensor
            Predictions of cell-type proportions.

        """
        self.model.eval()
        X = self.features
        adj = self.adj
        fX = self.model(X, adj)
        pred = F.softmax(fX, dim=1)
        return pred

    def score(self, pred, true_prop, score_metric="ce"):
        """Model performance score.

        Parameters
        ----------
        pred
            Predicted cell-type proportions.
        true_prop
            True cell-type proportions.
        score_metric
            Metric used to assess prediction performance.

        Returns
        -------
        loss : float
            Loss between predicted and true labels.

        """
        true_prop = true_prop.to(self.device)
        self.model.eval()
        if score_metric == "ce":
            loss = F.cross_entropy(pred, true_prop)
        elif score_metric == "mse":
            loss = ((pred / torch.sum(pred, 1, keepdims=True) -
                     true_prop / torch.sum(true_prop, 1, keepdims=True))**2).mean()
        return loss.detach().item()


def dropout_layer(x, dropout):
    """Dropout layer.

    Parameters
    ----------
    x
        input to dropout layer.
    dropout : float
        dropout rate (between 0 and 1).

    Returns
    -------
    out : torch tensor
        dropout output.

    """
    if x.is_sparse:  # sparse input features
        out = sparse_dropout(x, dropout)
        return out
    else:
        out = F.dropout(x, dropout)
        return out


def sparse_dropout(x, dropout):
    """Sparse dropout.

    Parameters
    ----------
    x
        Input to dropout layer.
    dropout : float
        Dropout rate (between 0 and 1).

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


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking.

    Parameters
    ----------
    preds:
        Cell-type proportion predictions from dstg model.
    labels :
        True cell-type proportion labels.
    mask :
        Mask to indicate which samples to use in computing loss.

    Returns
    -------
    loss : float
        Cross entropy loss between true and predicted cell-type proportions (mean reduced).

    """
    loss = F.cross_entropy(preds, labels, reduction="none")
    if mask is None:
        loss = loss.mean()
    else:
        loss = (loss * mask.float()).sum() / mask.sum()
    return loss
