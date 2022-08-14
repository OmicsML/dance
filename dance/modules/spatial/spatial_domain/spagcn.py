"""Reimplementation of SpaGCN.

Extended from https://github.com/jianhuupenn/SpaGCN

Reference
----------
Hu, Jian, et al. "SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and
spatially variable genes by graph convolutional network." Nature methods 18.11 (2021): 1342-1351.

"""

import math
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from models import GraphConvolution
from torch.nn.parameter import Parameter

from dance import utils


def refine(sample_id, pred, dis, shape="hexagon"):
    """An optional refinement step for the clustering result. In this step, SpaGCN
    examines the domain assignment of each spot and its surrounding spots. For a given
    spot, if more than half of its surrounding spots are assigned to a different domain,
    this spot will be relabeled to the same domain as the major label of its surrounding
    spots.

    Parameters
    ----------
    sample_id :
        sample id
    pred :
        initial prediction
    dis :
        graph structure
    shape : str optional
        by default as "hexagon"


    Returns
    -------
    refined_pred :
        refined prediction.

    """
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
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
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
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
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SimpleGCDEC(nn.Module):
    """Basic model used in SpaGCN training.

    Parameters
    ----------
    nfeat : int
        input feature dimension

    nhid : int
        output feature dimension

    alpha : float optional
        alpha, by default as 0.2

    """

    def __init__(self, nfeat, nhid, alpha=0.2):
        super().__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        #self.mu determined by the init method
        self.alpha = alpha

    def forward(self, x, adj):
        """forward function.

        Parameters
        ----------
        x :
            node features.
        adj :
            adjacent matrix.


        Returns
        -------
        x :
            the output of graph convolution layer.
        q :
            the probability of assigning cell i to cluster j.

        """
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        """objective function as a Kullback–Leibler (KL) divergence loss.

        Parameters
        ----------
        p :
            target distribution.
        q :
            the probability of assigning cell i to cluster j.


        Returns
        -------
        loss :
            Kullback–Leibler (KL) divergence loss.

        """

        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        """generate an auxiliary target distribution based on q the probability of
        assigning cell i to cluster j.

        Parameters
        ----------
        q :
            the probability of assigning cell i to cluster j.

        Returns
        -------
        p :
            target distribution.

        """
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, adj, lr=0.001, max_epochs=5000, update_interval=3, trajectory_interval=50, weight_decay=5e-4,
            opt="sgd", init="louvain", n_neighbors=10, res=0.4, n_clusters=10, init_spa=True, tol=1e-3, device="cuda"):
        """fit function for model training.

        Parameters
        ----------
        X :
            node features.
        adj :
            adjacent matrix.
        lr : float optional
            learning rate.
        max_epochs : int optional
            max epochs.
        update_interval: int optional
            interval for update
        trajectory_interval: int optional
            trajectory interval
        weight_decay : float optional
            weight decay.
        opt : str optional
            optimization.
        init : str optional
            "louvain" or "kmeans".
        n_neighbors : int optional
            the number of neighbors used in louvain.
        res : float optional
            used for louvain .
        n_clusters : int optional
            the number of clusters usedd in kmeans.
        init_spa : bool optional
            initialize spatial.
        tol : float optional
            tolerant value for searching l.

        Returns
        -------
        None.

        """
        self.trajectory = []
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features = self.gc(torch.FloatTensor(X), torch.FloatTensor(adj))
        #----------------------------------------------------------------
        if init == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                #------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
        elif init == "louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata = sc.AnnData(features.detach().numpy())
            else:
                adata = sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            # sc.tl.louvain(adata,resolution=res)
            sc.tl.leiden(adata, resolution=res, key_added='louvain')

            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        # judge have or no cuda device in torch
        if torch.cuda.is_available():
            if device == "cuda":
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.mu.data.copy_(torch.Tensor(cluster_centers))

        # copy data and model in cuda
        self = self.to(device)
        X = X.to(device)
        adj = adj.to(device)

        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(X, adj)
                p = self.target_distribution(q).data
            if epoch % 10 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            #Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.detach().cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break

        # recover model and data in cpu
        self = self.cpu()
        X = X.cpu()
        adj = adj.cpu()

    def fit_with_init(self, X, adj, init_y, lr=0.001, max_epochs=5000, update_interval=1, weight_decay=5e-4, opt="sgd"):
        """Initializing cluster centers with kmeans."""
        print("Initializing cluster centers with kmeans.")
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

        # judge have or no cuda device in torch
        if torch.cuda.is_available():
            if device == "cuda":
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # copy data and model in cuda
        self = self.to(device)
        X = X.to(device)
        adj = adj.to(device)

        self.train()
        for epoch in range(max_epochs):
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
        # recover model and data in cpu
        self = self.cpu()
        X = X.cpu()
        adj = adj.cpu()

    def predict(self, X, adj):
        """transform to float tensor."""
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
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, adj, lr=0.001, max_epochs=10, update_interval=5, weight_decay=5e-4, opt="sgd", init="louvain",
            n_neighbors=10, res=0.4):
        self.trajectory = []
        print("Initializing cluster centers with kmeans.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(torch.FloatTensor(X), torch.FloatTensor(adj))
        #----------------------------------------------------------------
        if init == "kmeans":
            #Kmeans only use exp info, no spatial
            #kmeans = KMeans(self.n_clusters, n_init=20)
            #y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
            #Kmeans use exp and spatial
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().numpy())
        elif init == "louvain":
            adata = sc.AnnData(features.detach().numpy())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
        #----------------------------------------------------------------
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(X, adj)
                p = self.target_distribution(q).data
            if epoch % 100 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

    def fit_with_init(self, X, adj, init_y, lr=0.001, max_epochs=10, update_interval=1, weight_decay=5e-4, opt="sgd"):
        print("Initializing cluster centers with kmeans.")
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
        for epoch in range(max_epochs):
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


class SpaGCN:
    """SpaGCN class.

    Parameters
    ----------
    l : float
        the parameter to control percentage p

    """

    def __init__(self, l=None):
        super().__init__()
        self.l = l or None
        self.res = None

    def search_l(self, p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
        """Search best l.

        Parameters
        ----------
        p : float
            percentage.
        adj :
            adjacent matrix.
        start : float optional
            starting value for searching l.
        end : float optional
            ending value for searching l.
        tol : float optional
            tolerant value for searching l.
        max_run : int optional
            Max runs.

        Returns
        -------
        l : float
            best l, the parameter to control percentage p.

        """
        l = utils.search_l(p, adj, start, end, tol, max_run)
        return l

    def set_l(self, l):
        """set l.

        Parameters
        ----------
        l : float
            the parameter to control percentage p.

        Returns
        -------
        None.

        """
        self.l = l

    def search_set_res(self, adata, adj, l, target_num, start=0.4, step=0.1, tol=5e-3, lr=0.05, max_epochs=10,
                       r_seed=100, t_seed=100, n_seed=100, max_run=10):
        """search res.

        res: Resolution in the initial Louvain's Clustering methods.

        """
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        res = start
        print("Start at res = ", res, "step = ", step)
        clf = SpaGCN()
        clf.set_l(l)
        clf.fit(adata, adj, init_spa=True, init="louvain", res=res, tol=tol, lr=lr, max_epochs=max_epochs)
        y_pred, _ = clf.predict()
        old_num = len(set(y_pred))
        print("Res = ", res, "Num of clusters = ", old_num)
        run = 0
        while old_num != target_num:
            random.seed(r_seed)
            torch.manual_seed(t_seed)
            np.random.seed(n_seed)
            old_sign = 1 if (old_num < target_num) else -1
            clf = SpaGCN()
            clf.set_l(l)
            clf.fit(adata, adj, init_spa=True, init="louvain", res=res + step * old_sign, tol=tol, lr=lr,
                    max_epochs=max_epochs)
            y_pred, _ = clf.predict()
            new_num = len(set(y_pred))
            print("Res = ", res + step * old_sign, "Num of clusters = ", new_num)
            if new_num == target_num:
                res = res + step * old_sign
                print("recommended res = ", str(res))
                return res
            new_sign = 1 if (new_num < target_num) else -1
            if new_sign == old_sign:
                res = res + step * old_sign
                print("Res changed to", res)
                old_num = new_num
            else:
                step = step / 2
                print("Step changed to", step)
            if run > max_run:
                print("Exact resolution not found")
                print("Recommended res = ", str(res))
                return res
            run += 1
        print("recommended res = ", str(res))
        self.res = res
        return res

    def fit(
            self,
            adata,
            adj,
            num_pcs=50,
            lr=0.005,
            max_epochs=2000,
            weight_decay=0,
            opt="admin",
            init_spa=True,
            init="louvain",  #louvain or kmeans
            n_neighbors=10,  #for louvain
            n_clusters=None,  #for kmeans
            res=0.4,  #for louvain
            tol=1e-3):
        """fit function for model training.

        Parameters
        ----------
        adata :
            input data.
        adj :
            adjacent matrix.
        num_pcs : int
            the number of component used in PCA.
        lr : float
            learning rate.
        max_epochs : int
            max epochs.
        weight_decay : float
            weight decay.
        opt : str
            optimization.
        init_spa : bool
            initialize spatial.
        init : str
            "louvain" or "kmeans".
        n_neighbors : int
            the number of neighbors used in louvain.
        n_clusters : int
            the number of clusters usedd in kmeans.
        res : float
            used for louvain .
        tol : float
            tolerant value for searching l.

        Returns
        -------
        None.

        """
        self.num_pcs = num_pcs
        self.res = res
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.opt = opt
        self.init_spa = init_spa
        self.init = init
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.res = res
        self.tol = tol
        assert adata.shape[0] == adj.shape[0] == adj.shape[1]
        pca = PCA(n_components=self.num_pcs)
        if issparse(adata.X):
            pca.fit(adata.X.A)
            embed = pca.transform(adata.X.A)
        else:
            pca.fit(adata.X)
            embed = pca.transform(adata.X)
        ###------------------------------------------###
        if self.l is None:
            raise ValueError('l should be set before fitting the model!')
        adj_exp = np.exp(-1 * (adj**2) / (2 * (self.l**2)))
        #----------Train model----------
        self.model = SimpleGCDEC(embed.shape[1], embed.shape[1])
        self.model.fit(embed, adj_exp, lr=self.lr, max_epochs=self.max_epochs, weight_decay=self.weight_decay,
                       opt=self.opt, init_spa=self.init_spa, init=self.init, n_neighbors=self.n_neighbors,
                       n_clusters=self.n_clusters, res=self.res, tol=self.tol)
        self.embed = embed
        self.adj_exp = adj_exp

    def predict(self):
        """prediction function.

        Parameters
        ----------

        Returns
        -------
        y_pred : numpy
            predicted label.
        prob : numpy
            predicted probability.

        """
        z, q = self.model.predict(self.embed, self.adj_exp)
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        self.y_pred = y_pred
        # Max probability plot
        prob = q.detach().numpy()
        return y_pred, prob

    def score(self, y_true):
        """score function to get score of prediction.

        Parameters
        ----------
        y_true :
            ground truth label.

        Returns
        -------
        score : float
            metric eval score.

        """
        from sklearn.metrics.cluster import adjusted_rand_score
        score = adjusted_rand_score(y_true, self.y_pred)
        print("ARI {}".format(adjusted_rand_score(y_true, self.y_pred)))
        return score
