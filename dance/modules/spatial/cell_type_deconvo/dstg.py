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

from dance.transforms import (AnnDataTransform, Compose, FilterGenesCommon, PseudoMixture, RemoveSplit, ScaleFeature,
                              SetConfig)
from dance.transforms.graph import DSTGraph
from dance.typing import LogLevel
from dance.utils import get_device


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907.

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

    def __init__(self, in_features, out_features, support, bias=False):
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


class DSTG:
    """DSTG cell-type deconvolution model.

    Parameters
    ----------
    nhid : int
        Number of units in the hidden layer (graph convolution).
    bias : boolean optional
        Include bias term, default False.
    dropout : float optional
        Dropout rate, default 0.
    device : str
        Computation device.

    """

    def __init__(self, nhid=32, bias=False, dropout=0, device="cpu"):
        super().__init__()
        self.nhid = nhid
        self.bias = bias
        self.dropout = dropout
        self.device = get_device(device)

    @staticmethod
    def preprocessing_pipeline(
        n_pseudo: int = 500,
        n_top_genes: int = 2000,
        hvg_flavor: str = "seurat",
        k_filter: int = 200,
        num_cc: int = 30,
        log_level: LogLevel = "INFO",
    ):
        return Compose(
            FilterGenesCommon(split_keys=["ref", "test"], log_level="INFO"),
            PseudoMixture(n_pseudo=n_pseudo, out_split_name="pseudo"),
            RemoveSplit(split_name="ref", log_level="INFO"),
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            AnnDataTransform(sc.pp.highly_variable_genes, flavor=hvg_flavor, n_top_genes=n_top_genes, batch_key="batch",
                             subset=True),
            ScaleFeature(split_names="ALL", mode="standardize"),
            DSTGraph(k_filter=k_filter, num_cc=num_cc, ref_split="pseudo", inf_split="test"),
            SetConfig({
                "feature_channel": [None, "DSTGraph"],
                "feature_channel_type": ["X", "obsp"],
                "label_channel": "cell_type_portion",
            }),
            log_level=log_level,
        )

    def _init_model(self, dim_in, dim_out):
        """Initialize GCN model."""
        self.model = GCN(dim_in, self.nhid, dim_out, self.bias, self.dropout).to(self.device)

    def fit(self, x, adj, y, train_mask, lr=0.005, max_epochs=50, weight_decay=0):
        """Fit function for model training.

        Parameters
        ----------
        x : torch.Tensor
            Gene expression feature (spot x gene).
        adj : torch.Tensor
            Spot similarity graph.
        y : torch.Tensor
            Cell-type portions (spot x cell-type).
        train_mask : torch.Tensor
            Mask indicating which spots (rows) should be used for training.
        lr : float optional
            Learning rate.
        max_epochs : int optional
            Maximum number of epochs to train.
        weight_decay : float optional
            Weight decay parameter for optimization (Adam).

        """
        x = x.to(self.device)
        adj = adj.to(self.device)
        y = y.to(self.device)
        train_mask = train_mask.to(self.device)
        self._init_model(x.shape[1], y.shape[1])

        model = self.model
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(max_epochs):
            t = time.time()

            y_pred = model(x, adj)
            loss = masked_softmax_cross_entropy(y_pred, y, train_mask)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch: {epoch + 1:04d}, train_loss={loss:.5f}, time={time.time() - t:.5f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        self.pred = F.softmax(self.model(x, adj), dim=-1)

    def predict(self):
        """Prediction function.

        Returns
        -------
        pred : torch tensor
            Predictions of cell-type proportions.

        """
        return self.pred

    def fit_and_predict(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.predict()

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
