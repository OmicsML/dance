"""Reimplementation of SpaGCN.

Extended from https://github.com/jianhuupenn/SpaGCN

Reference
----------
Hu, Jian, et al. "SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and
spatially variable genes by graph convolutional network." Nature methods 18.11 (2021): 1342-1351.

"""
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter

from dance import logger
from dance.modules.base import BaseClusteringMethod
from dance.transforms import AnnDataTransform, CellPCA, Compose, FilterGenesMatch, SetConfig
from dance.transforms.graph import SpaGCNGraph, SpaGCNGraph2D
from dance.typing import LogLevel


def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj**2) / (2 * (l**2)))
    return np.mean(np.sum(adj_exp, 1)) - 1


def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run = 0
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        logger.info("l not found, try smaller start point.")
        return None
    elif p_high < p - tol:
        logger.info("l not found, try bigger end point.")
        return None
    elif np.abs(p_low - p) <= tol:
        logger.info(f"recommended l: {start}")
        return start
    elif np.abs(p_high - p) <= tol:
        logger.info(f"recommended l: {end}")
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        logger.info(f"Run {run}: l [{start}, {end}], p [{p_low}, {p_high}]")
        if run > max_run:
            logger.info(f"Exact l not found, closest values are:\nl={start}: p={p_low}\nl={end}: p={p_high}")
            return None
        mid = (start + end) / 2
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            logger.info(f"recommended l: {mid}")
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid
    return None


def refine(sample_id, pred, dis, shape="hexagon"):
    """An optional refinement step for the clustering result. In this step, SpaGCN
    examines the domain assignment of each spot and its surrounding spots. For a given
    spot, if more than half of its surrounding spots are assigned to a different domain,
    this spot will be relabeled to the same domain as the major label of its surrounding
    spots.

    Parameters
    ----------
    sample_id
        Sample id.
    pred
        Initial prediction.
    dis
        Graph structure.
    shape : str
        Shape parameter.

    Returns
    -------
    refined_pred
        Refined predictions.

    """
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        logger.info("Shape not recognized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"


class SimpleGCDEC(nn.Module):
    """Basic model used in SpaGCN training.

    Parameters
    ----------
    nfeat : int
        Input feature dimension.
    nhid : int
        Output feature dimension.
    alpha : float optional
        Alphat parameter.

    """

    def __init__(self, nfeat, nhid, alpha=0.2, device="cpu"):
        super().__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        # self.mu is determined by the init method
        self.alpha = alpha
        self.device = device

    def forward(self, x, adj):
        """Forward function."""
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        """Objective function as a Kullback–Leibler (KL) divergence loss."""

        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        """Generate an auxiliary target distribution based on q the probability of
        assigning cell i to cluster j.

        Parameters
        ----------
        q
            The probability of assigning cell i to cluster j.

        Returns
        -------
        p
            Target distribution.

        """
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, adj, lr=0.001, epochs=5000, update_interval=3, trajectory_interval=50, weight_decay=5e-4,
            opt="sgd", init="louvain", n_neighbors=10, res=0.4, n_clusters=10, init_spa=True, tol=1e-3):
        """Fit function for model training.

        Parameters
        ----------
        X
            Node features.
        adj
            Adjacent matrix.
        lr : float
            Learning rate.
        epochs : int
            Maximum number of epochs.
        update_interval : int
            Interval for update.
        trajectory_interval: int
            Trajectory interval.
        weight_decay : float
            Weight decay.
        opt : str
            Optimizer.
        init : str
            "louvain" or "kmeans".
        n_neighbors : int
            The number of neighbors used in louvain.
        res : float
            Used for louvain.
        n_clusters : int
            The number of clusters usedd in kmeans.
        init_spa : bool
            Initialize spatial.
        tol : float
            Tolerant value for searching l.

        """
        self.trajectory = []
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features = self.gc(torch.FloatTensor(X), torch.FloatTensor(adj))
        # ---------------------------------------------------------------------
        if init == "kmeans":
            logger.info("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                # Kmeans using both expression and spatial information
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                # Kmeans using only expression information
                y_pred = kmeans.fit_predict(X)  # use X as numpy
        elif init == "louvain":
            logger.info(f"Initializing cluster centers with louvain, resolution = {res}")
            if init_spa:
                adata = sc.AnnData(features.detach().numpy())
            else:
                adata = sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            # sc.tl.louvain(adata,resolution=res)
            sc.tl.leiden(adata, resolution=res, key_added="louvain")

            y_pred = adata.obs["louvain"].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        # ---------------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))

        # Copy data and model in cuda
        device = self.device
        self = self.to(device)
        X = X.to(device)
        adj = adj.to(device)

        self.train()
        for epoch in range(epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(X, adj)
                p = self.target_distribution(q).data
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}")
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.detach().cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                logger.info(f"delta_label {delta_label} < tol {tol}")
                logger.info("Reach tolerance threshold. Stopping training.")
                logger.info(f"Total epoch: {epoch}")
                break

        # Recover model and data in cpu
        self = self.cpu()
        X = X.cpu()
        adj = adj.cpu()

    def fit_with_init(self, X, adj, init_y, lr=0.001, epochs=5000, update_interval=1, weight_decay=5e-4, opt="sgd"):
        """Initializing cluster centers with kmeans."""
        logger.info("Initializing cluster centers with kmeans.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        features, _ = self.forward(X, adj)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(init_y, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))

        # Copy data and model in cuda
        device = self.device
        self = self.to(device)
        X = X.to(device)
        adj = adj.to(device)

        self.train()
        for epoch in range(epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X), torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X = torch.FloatTensor(X)
            adj = torch.FloatTensor(adj)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

        # Recover model and data in cpu
        self = self.cpu()
        X = X.cpu()
        adj = adj.cpu()

    def predict(self, X, adj):
        """Transform to float tensor."""
        z, q = self(torch.FloatTensor(X), torch.FloatTensor(adj))
        return z, q


# Not used in SpaGCN
class GC_DEC(nn.Module):

    def __init__(self, nfeat, nhid1, nhid2, n_clusters=None, dropout=0.5, alpha=0.2):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.mu = Parameter(torch.Tensor(n_clusters, nhid2))
        self.n_clusters = n_clusters
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc2(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-6)
        q = q**(self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):

        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, adj, lr=0.001, epochs=10, update_interval=5, weight_decay=5e-4, opt="sgd", init="louvain",
            n_neighbors=10, res=0.4):
        self.trajectory = []
        logger.info("Initializing cluster centers with kmeans.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(torch.FloatTensor(X), torch.FloatTensor(adj))
        # ---------------------------------------------------------------------
        if init == "kmeans":
            # Kmeans using only expression information
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().numpy())
        elif init == "louvain":
            # Louvain using only expression information
            adata = sc.AnnData(features.detach().numpy())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs["louvain"].astype(int).to_numpy()
        # ---------------------------------------------------------------------
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(X, adj)
                p = self.target_distribution(q).data
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}")
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

    def fit_with_init(self, X, adj, init_y, lr=0.001, epochs=10, update_interval=1, weight_decay=5e-4, opt="sgd"):
        logger.info("Initializing cluster centers with kmeans.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        features, _ = self.forward(X, adj)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(init_y, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X), torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X = torch.FloatTensor(X)
            adj = torch.FloatTensor(adj)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        z, q = self(torch.FloatTensor(X), torch.FloatTensor(adj))
        return z, q


class SpaGCN(BaseClusteringMethod):
    """SpaGCN class.

    Parameters
    ----------
    l : float
        the parameter to control percentage p

    """

    def __init__(self, l=None, device="cpu"):
        self.l = l
        self.res = None
        self.device = device

    @staticmethod
    def preprocessing_pipeline(alpha: float = 1, beta: int = 49, dim: int = 50, log_level: LogLevel = "INFO"):
        return Compose(
            FilterGenesMatch(prefixes=["ERCC", "MT-"]),
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            SpaGCNGraph(alpha=alpha, beta=beta),
            SpaGCNGraph2D(),
            CellPCA(n_components=dim),
            SetConfig({
                "feature_channel": ["CellPCA", "SpaGCNGraph", "SpaGCNGraph2D"],
                "feature_channel_type": ["obsm", "obsp", "obsp"],
                "label_channel": "label",
                "label_channel_type": "obs"
            }),
            log_level=log_level,
        )

    def search_l(self, p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
        """Search best l.

        Parameters
        ----------
        p : float
            Percentage.
        adj :
            Adjacent matrix.
        start : float
            Starting value for searching l.
        end : float
            Ending value for searching l.
        tol : float
            Tolerant value for searching l.
        max_run : int
            Maximum number of runs.

        Returns
        -------
        l : float
            best l, the parameter to control percentage p.

        """
        l = search_l(p, adj, start, end, tol, max_run)
        return l

    def set_l(self, l):
        """Set l.

        Parameters
        ----------
        l : float
            The parameter to control percentage p.

        """
        self.l = l

    def search_set_res(self, x, l, target_num, start=0.4, step=0.1, tol=5e-3, lr=0.05, epochs=10, max_run=10):
        """Search for optimal resolution parameter."""
        res = start
        logger.info(f"Start at {res = :.4f}, {step = :.4f}")
        clf = SpaGCN(l)
        y_pred = clf.fit_predict(x, init_spa=True, init="louvain", res=res, tol=tol, lr=lr, epochs=epochs)
        old_num = len(set(y_pred))
        logger.info(f"Res = {res:.4f}, Num of clusters = {old_num}")
        run = 0
        while old_num != target_num:
            old_sign = 1 if (old_num < target_num) else -1
            clf = SpaGCN(l)
            y_pred = clf.fit_predict(x, init_spa=True, init="louvain", res=res + step * old_sign, tol=tol, lr=lr,
                                     epochs=epochs)
            new_num = len(set(y_pred))
            logger.info(f"Res = {res + step * old_sign:.3e}, Num of clusters = {new_num}")
            if new_num == target_num:
                res = res + step * old_sign
                logger.info(f"recommended res = {res:.4f}")
                return res
            new_sign = 1 if (new_num < target_num) else -1
            if new_sign == old_sign:
                res = res + step * old_sign
                logger.info(f"Res changed to {res}")
                old_num = new_num
            else:
                step = step / 2
                logger.info(f"Step changed to {step:.4f}")
            if run > max_run:
                logger.info(f"Exact resolution not found. Recommended res = {res:.4f}")
                return res
            run += 1
        logger.info("Recommended res = {res:.4f}")
        self.res = res
        return res

    def calc_adj_exp(self, adj: np.ndarray) -> np.ndarray:
        adj_exp = np.exp(-1 * (adj**2) / (2 * (self.l**2)))
        return adj_exp

    def fit(self, x, y=None, *, num_pcs=50, lr=0.005, epochs=2000, weight_decay=0, opt="admin", init_spa=True,
            init="louvain", n_neighbors=10, n_clusters=None, res=0.4, tol=1e-3):
        """Fit function for model training.

        Parameters
        ----------
        embed
            Input data.
        adj
            Adjacent matrix.
        num_pcs : int
            The number of component used in PCA.
        lr : float
            Learning rate.
        epochs : int
            Maximum number of epochs.
        weight_decay : float
            Weight decay.
        opt : str
            Optimizer.
        init_spa : bool
            Initialize spatial.
        init : str
            "louvain" or "kmeans".
        n_neighbors : int
            The number of neighbors used by Louvain.
        n_clusters : int
            The number of clusters usedd by kmeans.
        res : float
            The resolution parameter used by Louvain.
        tol : float
            Oolerant value for searching l.

        """
        embed, adj = x
        self.num_pcs = num_pcs
        self.res = res
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.opt = opt
        self.init_spa = init_spa
        self.init = init
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.res = res
        self.tol = tol
        if self.l is None:
            raise ValueError("l should be set before fitting the model!")

        self.model = SimpleGCDEC(embed.shape[1], embed.shape[1], device=self.device)
        adj_exp = self.calc_adj_exp(adj)
        self.model.fit(embed, adj_exp, lr=self.lr, epochs=self.epochs, weight_decay=self.weight_decay, opt=self.opt,
                       init_spa=self.init_spa, init=self.init, n_neighbors=self.n_neighbors, n_clusters=self.n_clusters,
                       res=self.res, tol=self.tol)

    def predict_proba(self, x):
        """Prediction function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The predicted labels and the predicted probabilities.

        """
        embed, adj = x
        adj_exp = self.calc_adj_exp(adj)
        _, pred_prob = self.model.predict(embed, adj_exp)
        return pred_prob

    def predict(self, x):
        """Prediction function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The predicted labels and the predicted probabilities.

        """
        pred_prob = self.predict_proba(x)
        pred = torch.argmax(pred_prob, dim=1).data.cpu().numpy()
        return pred
