import os
import random
import sys
import time
from collections import Counter

import anndata as ad
import dgl
import numba
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn.neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from munkres import Munkres
from scipy.spatial import distance_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.backends import cudnn

from dance.modules.base import BaseClusteringMethod
from dance.transforms import Compose, HighlyVariableGenesRawCount, NormalizeTotalLog1P, PrefilterGenes, SetConfig
from dance.transforms.base import BaseTransform
from dance.transforms.graph import CalSpatialNet
from dance.typing import LogLevel

# from torch_geometric.data import Data

"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i])**2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def calculate_adj_matrix(adata):
    x = adata.obs["array_row"]
    y = adata.obs["array_col"]
    X = np.array([x, y]).T.astype(np.float32)
    adj = pairwise_distance(X)
    return adj


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NB:

    def __init__(self, theta=None, scale_factor=1.0):
        super().__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
            y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):

    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


def compute_joint(view1, view2):
    """Compute the joint probability matrix P."""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2)**2)


def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency."""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    #     Works with pytorch <= 1.2
    #     p_i_j[(p_i_j < EPS).data] = EPS
    #     p_j[(p_j < EPS).data] = EPS
    #     p_i[(p_i < EPS).data] = EPS

    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss * -1


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def regularization_loss(emb, adj):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    loss = torch.mean((mat - adj)**2)
    return loss


def refine_label(adata, radius=50, key='cluster'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position.to_numpy(), position.to_numpy(), metric='euclidean')
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def munkres_newlabel(y_true, y_pred):
    """\ Kuhn-Munkres algorithm to achieve mapping from cluster labels to ground truth
    label.

    Parameters
    ----------
    y_true
        ground truth label
    y_pred
        cluster labels

    Returns
    -------
    mapping label

    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return 0, 0, 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    print('Counter(new_predict)\n', Counter(new_predict))
    print('Counter(y_true)\n', Counter(y_true))

    return new_predict


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', Spatial_uns="Spatial_Net"):
    """Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']

    """

    print('------Calculating spatial graph...')
    assert (model in ['Radius', 'KNN'])
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[
        Spatial_Net['Distance'] > 0,
    ]
    id_cell_trans = dict(zip(
        range(coor.shape[0]),
        np.array(coor.index),
    ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    adata.uns[Spatial_uns] = Spatial_Net


class TransferData(BaseTransform):
    """Transfer spatial network data to graph format.

    Parameters
    ----------
    spatial_uns
        Key for spatial networks in adata.uns. Default is "Spatial_Net".

    """

    _DISPLAY_ATTRS = ("spatial_uns", )

    def __init__(self, spatial_uns="Spatial_Net", out=None, log_level="WARNING"):
        super().__init__(out=out, log_level=log_level)
        self.spatial_uns = spatial_uns

    def __call__(self, data):
        """Transfer spatial network data to graph format."""
        adata = data.data

        G_df = adata.uns[self.spatial_uns].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

        G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
        G = G + sp.eye(G.shape[0])

        # Return the results as in the original function
        if hasattr(adata.X, 'todense'):
            features = adata.X.todense()
        else:
            features = adata.X
        return G, features


def Transfer_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    return G, adata.X.todense()


class NB:

    def __init__(self, theta=None, scale_factor=1.0):
        super().__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
            y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):

    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


class decoder(torch.nn.Module):

    def __init__(self, nfeat, nhid1, nhid2):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid1), torch.nn.BatchNorm1d(nhid1), torch.nn.ReLU())
        self.pi = torch.nn.Linear(nhid1, nhid2)
        self.disp = torch.nn.Linear(nhid1, nhid2)
        self.mean = torch.nn.Linear(nhid1, nhid2)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]

    def refine_label(adata, radius=50, key='cluster'):
        n_neigh = radius
        new_type = []
        old_type = adata.obs[key].values

        # calculate distance
        position = adata.obsm['spatial']
        distance = ot.dist(position, position, metric='euclidean')
        n_cell = distance.shape[0]

        for i in range(n_cell):
            vec = distance[i, :]
            index = vec.argsort()
            neigh_type = []
            for j in range(1, n_neigh + 1):
                neigh_type.append(old_type[index[j]])
            max_type = max(neigh_type, key=neigh_type.count)
            new_type.append(max_type)

        new_type = [str(i) for i in list(new_type)]
        # adata.obs['label_refined'] = np.array(new_type)

        return new_type


class GAT(nn.Module):

    def __init__(self, g, H, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope):
        super().__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.ZINB = decoder(num_hidden * heads[0], H, in_dim)

        self.gat_layers.append(
            GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(num_hidden * heads[l - 1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, False,
                        self.activation))

    def forward(self, inputs):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[l](self.g, temp)
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])

        heads_tensor = torch.cat(heads, axis=1)
        [pi, disp, mean] = self.ZINB(heads_tensor)
        return heads, pi, disp, mean


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj, mean: bool = True, tau: float = 1.0,
                     hidden_norm: bool = True):
    l1 = nei_con_loss(z1, z2, tau, adj, hidden_norm)
    l2 = nei_con_loss(z2, z1, tau, adj, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret


def multihead_contrastive_loss(heads, adj, tau: float = 1.0):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss + contrastive_loss(heads[0], heads[i], adj, tau=tau)
    return loss / (len(heads) - 1)


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


# --- 修改 1: 支持 Batch 的对比损失函数 ---


def multihead_contrastive_loss(heads, adj, nei_count, tau: float = 1.0):
    """
    heads: List of tensors, 每个 tensor 是 (Batch_Size, Hidden_Dim)
    adj: (Batch_Size, Batch_Size) 稠密矩阵
    nei_count: (Batch_Size, ) 节点的度数
    """
    loss = torch.tensor(0, dtype=torch.float32, device=heads[0].device, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss + nei_con_loss(heads[0], heads[i], tau, adj, nei_count)
    return loss / (len(heads) - 1)


def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, nei_count, hidden_norm: bool = True):
    '''
    neighbor contrastive loss (Batch Version)
    z1, z2: (Batch_Size, Dim)
    adj: (Batch_Size, Batch_Size) Dense Tensor
    nei_count: (Batch_Size)
    '''
    # 移除自环 (Batch 内部的对角线)
    adj = adj - torch.diag_embed(torch.diag(adj))
    adj[adj > 0] = 1  # 确保是二值的

    # 这里我们只计算 Batch 内的相似度，因此显存占用极小 (2048^2)
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    # 计算 Loss
    # 注意：这里的 sum(1) 仅是 Batch 内的求和，作为全图的近似
    denom = intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag()

    # 避免分母为0
    denom = torch.clamp(denom, min=1e-6)

    loss = (inter_view_sim.diag() + (intra_view_sim * adj).sum(1) + (inter_view_sim * adj).sum(1)) / denom

    # 使用预先计算好的真实邻居数量进行归一化
    # 注意避免除以0
    nei_count = torch.clamp(nei_count, min=1.0)
    loss = loss / nei_count

    return -torch.log(torch.clamp(loss, min=1e-8)).mean()


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-", Gene3Pattern="mt-"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp3 = np.asarray([not str(name).startswith(Gene3Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2, id_tmp3)
    adata._inplace_subset_var(id_tmp)


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def refine_nearest_labels(adata, radius=50, key='label'):
    new_type = []
    df = adata.obsm['spatial']
    old_type = adata.obs[key].values
    df = pd.DataFrame(df, index=old_type)
    distances = distance_matrix(df, df)
    distances_df = pd.DataFrame(distances, index=old_type, columns=old_type)

    for index, row in distances_df.iterrows():
        # row[index] = np.inf
        nearest_indices = row.nsmallest(radius).index.tolist()
        # for i in range(1):
        #     nearest_indices.append(index)
        max_type = max(nearest_indices, key=nearest_indices.count)
        new_type.append(max_type)
        # most_common_element, most_common_count = find_most_common_elements(nearest_indices)
        # nearest_labels.append(df.loc[nearest_indices, 'label'].values)

    return [str(i) for i in list(new_type)]


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(
        np.float32
    )  #其思想是 按照(row_index, column_index, value)的方式存储每一个非0元素，所以存储的数据结构就应该是一个以三元组为元素的列表List[Tuple[int, int, int]]
    indices = torch.from_numpy(np.vstack(
        (sparse_mx.row, sparse_mx.col)).astype(np.int64))  #from_numpy()用来将数组array转换为张量Tensor vstack（）：按行在下边拼接
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def merge_anndatas(adata_list, batch_key="slide"):
    combined_adata = adata_list[0].concatenate(adata_list[1:], join='outer', batch_key=batch_key)
    spatial_nets = []
    for i, adata in enumerate(adata_list):
        if 'Spatial_Net' in adata.uns:
            spatial_net = adata.uns['Spatial_Net'].copy()
            spatial_net['Cell1'] = spatial_net['Cell1'] + '-' + str(i)
            spatial_net['Cell2'] = spatial_net['Cell2'] + '-' + str(i)
            spatial_nets.append(spatial_net)
    if spatial_nets:
        combined_spatial_net = pd.concat(spatial_nets, ignore_index=True)
        combined_adata.uns['Spatial_Net'] = combined_spatial_net
    return combined_adata


def get_adata(file_name="151507.h5ad"):
    adata = sc.read(file_name)
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    Cal_Spatial_Net(adata, rad_cutoff=150)


def train(adata, k=0, hidden_dims=3000, n_epochs=100, num_hidden=100, lr=0.00008, key_added='SpaGRA', a=0.1, b=1, c=0.5,
          radius=50, weight_decay=0.0001, random_seed=0, feat_drop=0.01, attn_drop=0.1, negative_slope=0.01, heads=4,
          method="kmeans", reso=1, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    set_seed(random_seed)
    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    adj, features = Transfer_Data(adata_Vars)
    g = dgl.from_scipy(adj)
    all_time = time.time()
    g = g.int().to(device)
    num_feats = features.shape[1]
    n_edges = g.number_of_edges()
    model = GAT(g, hidden_dims, 1, num_feats, num_hidden, [heads], F.elu, feat_drop, attn_drop, negative_slope)
    adj = torch.tensor(adj.todense()).to(device)
    features = torch.FloatTensor(features).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    coords = torch.tensor(adata.obsm['spatial']).float().to(device)
    sp_dists = torch.cdist(coords, coords, p=2)
    sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
    ari_max = 0
    model.train()
    for epoch in range(n_epochs):

        # model.train()
        optimizer.zero_grad()
        heads, pi, disp, mean = model(features)
        heads0 = torch.cat(heads, axis=1)
        # heads0 = heads[0]

        z_dists = torch.cdist(heads0, heads0, p=2)
        z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
        n_items = heads0.size(dim=0) * heads0.size(dim=0)
        reg_loss = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
        # reg_loss=0
        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features, mean, mean=True)
        loss = multihead_contrastive_loss(heads, adj, tau=10)
        total_loss = a * loss + b * reg_loss + c * zinb_loss

        total_loss.backward()
        optimizer.step()
        print("loss ", epoch, loss.item(), reg_loss.item(), zinb_loss.item())
        # kmeans = KMeans(n_clusters=k).fit(np.nan_to_num(heads0.cpu().detach()))
        if method == "kmeans":
            kmeans = KMeans(n_clusters=k, random_state=random_seed).fit(np.nan_to_num(heads0.cpu().detach()))
            idx = kmeans.labels_
            adata_Vars.obs['temp'] = idx
            obs_df = adata_Vars.obs.dropna()

            if 'Ground Truth' in obs_df.columns:
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['Ground Truth'])
                # print("ARI:",ari_res,"MAX ARI:",ari_max)
                if ari_res > ari_max:
                    ari_max = ari_res
                    idx_max = idx
                    mean_max = mean.to('cpu').detach().numpy()
                    emb_max = heads0.to('cpu').detach().numpy()
            else:
                idx_max = idx
                mean_max = mean.to('cpu').detach().numpy()
                emb_max = heads0.to('cpu').detach().numpy()

        if method == "louvain":
            adata_tmp = sc.AnnData(np.nan_to_num(heads0.cpu().detach()))
            sc.pp.neighbors(adata_tmp, n_neighbors=20, use_rep='X')
            sc.tl.louvain(adata_tmp, resolution=reso, random_state=0)
            idx = adata_tmp.obs['louvain'].astype(int).to_numpy()

    if method == "kmeans":
        adata.obs["cluster"] = idx_max.astype(str)
        adata.obsm["emb"] = emb_max
        adata.obsm['mean'] = mean_max

    if method == "louvain":
        adata.obs["cluster"] = idx.astype(str)
        emb = heads0.to('cpu').detach().numpy()
        adata.obsm["emb"] = emb  ######

    if radius != 0:
        nearest_new_type = refine_label(adata, radius=radius)
        adata.obs[key_added] = nearest_new_type
    else:
        adata.obs[key_added] = adata.obs["cluster"]
    # adata.obsm["emb"] = emb_max
    # adata.obsm['mean'] = mean_max
    model.eval()
    heads, pi, disp, mean = model(features)
    z = torch.cat(heads, axis=1)
    adata.obsm[key_added] = z.to('cpu').detach().numpy()
    pca = PCA(n_components=50, random_state=0)
    adata.obsm['emb_pca'] = pca.fit_transform(adata.obsm['emb'].copy())

    return adata


class SpaGRA(BaseClusteringMethod):
    """SpaGRA class for spatial domain identification using graph attention networks.

    Parameters
    ----------
    k : int, optional
        Number of clusters for clustering. Default is 0.
    hidden_dims : int, optional
        Hidden dimensions for the decoder. Default is 3000.
    n_epochs : int, optional
        Number of training epochs. Default is 100.
    num_hidden : int, optional
        Number of hidden units in GAT layers. Default is 100.
    lr : float, optional
        Learning rate. Default is 0.00008.
    key_added : str, optional
        Key for storing results in adata.obs. Default is 'SpaGRA'.
    a : float, optional
        Weight for contrastive loss. Default is 0.1.
    b : float, optional
        Weight for regularization loss. Default is 1.
    c : float, optional
        Weight for ZINB loss. Default is 0.5.
    radius : int, optional
        Radius for label refinement. Default is 50.
    weight_decay : float, optional
        Weight decay for optimizer. Default is 0.0001.
    random_seed : int, optional
        Random seed. Default is 0.
    feat_drop : float, optional
        Feature dropout rate. Default is 0.01.
    attn_drop : float, optional
        Attention dropout rate. Default is 0.1.
    negative_slope : float, optional
        Negative slope for LeakyReLU. Default is 0.01.
    heads : int, optional
        Number of attention heads. Default is 4.
    method : str, optional
        Clustering method, either "kmeans" or "louvain". Default is "kmeans".
    reso : float, optional
        Resolution for Louvain clustering. Default is 1.
    device : torch.device, optional
        Device for computation. Default is CUDA if available, else CPU.

    """

    def __init__(self, k=0, hidden_dims=1000, n_epochs=200, num_hidden=600, lr=0.00008, key_added='SpaGRA', a=2, b=1,
                 c=1, radius=0, weight_decay=0.00001, random_seed=0, feat_drop=0.02, attn_drop=0.01,
                 negative_slope=0.02, heads=4, method="louvain", reso=0.8,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.k = k
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.num_hidden = num_hidden
        self.lr = lr
        self.key_added = key_added
        self.a = a
        self.b = b
        self.c = c
        self.radius = radius
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.heads = heads
        self.method = method
        self.reso = reso
        self.device = device

        self.model = None
        self.adata = None

    @staticmethod
    def preprocessing_pipeline(log_level: LogLevel = "INFO"):
        transforms = []
        transforms.append(PrefilterGenes(min_cells=3))
        transforms.append(HighlyVariableGenesRawCount(n_top_genes=1000))
        transforms.append(NormalizeTotalLog1P())
        transforms.append(CalSpatialNet(rad_cutoff=150))
        transforms.append(SetConfig({"label_channel": "label", "label_channel_type": "obs"}))
        return Compose(*transforms, log_level=log_level)

    # @classmethod
    # def get_adata(cls, file_name="151507.h5ad"):
    #     """Load and preprocess AnnData object."""
    #     adata = sc.read(file_name)
    #     adata.var_names_make_unique()
    #     prefilter_genes(adata, min_cells=3)
    #     sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    #     sc.pp.normalize_per_cell(adata)
    #     sc.pp.log1p(adata)
    #     # Use the new CalSpatialNet class
    #     cal_spatial_net = CalSpatialNet(rad_cutoff=150)
    #     from dance.data import Data
    #     data = Data(adata)
    #     cal_spatial_net(data)
    #     return data.data

    def fit(self, adata):
        """Fit the SpaGRA model (Fully Batch-Optimized for OOM issues)."""
        set_seed(self.random_seed)
        adata.X = sp.csr_matrix(adata.X)

        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata

        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

        # Data Preparation
        from dance.data import Data
        data = Data(adata_Vars)
        transfer_data = TransferData()
        adj, features = transfer_data(data)  # adj here contains self-loops (eye)

        # 1. 构建 DGL 图用于 GAT (保持不变)
        g = dgl.from_scipy(adj)
        g = g.int().to(self.device)
        num_feats = features.shape[1]

        # 2. 预先计算所有节点的邻居数量 (在 CPU 上进行，避免 GPU OOM)
        # adj 已经包含了自环，所以 sum 结果至少是 1
        # nei_con_loss 逻辑中：nei_count = intra_nei + inter_nei + self_inter
        # 原逻辑近似于度数 * 2 + 1。我们直接预计算好。
        adj_no_eye = adj - sp.eye(adj.shape[0])
        raw_degree = np.array(adj_no_eye.sum(1)).flatten()
        # 根据原代码逻辑: nei_count = degree * 2 + 1
        all_nei_counts = torch.tensor(raw_degree * 2 + 1).float().to(self.device)

        self.model = GAT(g, self.hidden_dims, 1, num_feats, self.num_hidden, [self.heads], F.elu, self.feat_drop,
                         self.attn_drop, self.negative_slope)

        features = torch.FloatTensor(features).to(self.device)
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 3. 准备坐标数据
        coords = torch.tensor(adata.obsm['spatial'].values).float().to(self.device)
        ari_max = 0

        print(f"Start Training: N={adata.n_obs}, Batch_Size=2048")
        self.model.train()

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            heads, pi, disp, mean = self.model(features)
            heads0 = torch.cat(heads, axis=1)

            # --- 核心优化: 采样 ---
            # 每次随机选取 2048 个点进行 Loss 计算
            batch_size = 2048
            if heads0.shape[0] > batch_size:
                idx = torch.randperm(heads0.shape[0])[:batch_size]
            else:
                idx = torch.arange(heads0.shape[0])

            idx = idx.to(self.device)
            idx_cpu = idx.cpu().numpy()  # 用于切片 scipy 矩阵

            # 1. Batch Data 准备
            heads_batch = [h[idx] for h in heads]  # List slicing
            heads0_batch = heads0[idx]
            coords_batch = coords[idx]
            nei_count_batch = all_nei_counts[idx]

            # 2. 动态生成 Batch 的邻接矩阵 (从 Scipy 切片 -> Dense Tensor)
            # 这一步在 CPU 做切片，生成的矩阵只有 2048x2048，显存占用极小
            adj_batch_scipy = adj[idx_cpu, :][:, idx_cpu]
            adj_batch = torch.tensor(adj_batch_scipy.todense()).float().to(self.device)

            # 3. 计算正则化 Loss (Batch)
            z_dists_batch = torch.cdist(heads0_batch, heads0_batch, p=2)
            z_dists_batch = torch.div(z_dists_batch, torch.max(z_dists_batch) + 1e-6)

            sp_dists_batch = torch.cdist(coords_batch, coords_batch, p=2)
            sp_dists_batch = torch.div(sp_dists_batch, torch.max(sp_dists_batch) + 1e-6)

            reg_loss = torch.mean(torch.mul(1.0 - z_dists_batch, sp_dists_batch))

            # 4. 计算 Contrastive Loss (Batch)
            loss = multihead_contrastive_loss(heads_batch, adj_batch, nei_count_batch, tau=10)

            # 5. ZINB Loss (Global，因为这个计算不涉及 N*N 矩阵，通常不会 OOM)
            zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features, mean, mean=True)

            total_loss = self.a * loss + self.b * reg_loss + self.c * zinb_loss

            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Reg={reg_loss.item():.4f}, ZINB={zinb_loss.item():.4f}")

            # --- Clustering Logic (保持原样，增加频率限制以提速) ---
            if self.method == "kmeans" and (epoch % 20 == 0 or epoch == self.n_epochs - 1):
                # 如果显存依然紧张，这里可以将 data 移到 CPU
                data_for_clustering = np.nan_to_num(heads0.detach().cpu().numpy())
                kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=self.random_seed).fit(data_for_clustering)
                idx_res = kmeans.labels_
                adata_Vars.obs['temp'] = idx_res
                obs_df = adata_Vars.obs.dropna()

                if 'Ground Truth' in obs_df.columns:
                    ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['Ground Truth'])
                    if ari_res > ari_max:
                        ari_max = ari_res
                        idx_max = idx_res
                        mean_max = mean.detach().cpu().numpy()
                        emb_max = heads0.detach().cpu().numpy()
                else:
                    idx_max = idx_res
                    mean_max = mean.detach().cpu().numpy()
                    emb_max = heads0.detach().cpu().numpy()

            if self.method == "louvain" and epoch == self.n_epochs - 1:
                adata_tmp = sc.AnnData(np.nan_to_num(heads0.detach().cpu().numpy()))
                sc.pp.neighbors(adata_tmp, n_neighbors=20, use_rep='X')
                sc.tl.louvain(adata_tmp, resolution=self.reso, random_state=0)
                idx_louvain = adata_tmp.obs['louvain'].astype(int).to_numpy()

        # Store results
        if self.method == "kmeans":
            adata.obs["cluster"] = idx_max.astype(str)
            adata.obsm["emb"] = emb_max
            adata.obsm['mean'] = mean_max

        if self.method == "louvain":
            adata.obs["cluster"] = idx_louvain.astype(str)
            emb = heads0.to('cpu').detach().numpy()
            adata.obsm["emb"] = emb

        if self.radius != 0:
            nearest_new_type = refine_label(adata, radius=self.radius)
            adata.obs[self.key_added] = nearest_new_type
        else:
            adata.obs[self.key_added] = adata.obs["cluster"]

        self.model.eval()
        with torch.no_grad():
            heads, pi, disp, mean = self.model(features)
            z = torch.cat(heads, axis=1)
            adata.obsm[self.key_added] = z.to('cpu').detach().numpy()
            pca = PCA(n_components=50, random_state=0)
            adata.obsm['emb_pca'] = pca.fit_transform(adata.obsm['emb'].copy())

        self.adata = adata

    def predict(self, x=None):
        """Return clustering predictions."""
        if self.adata is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.adata.obs[self.key_added].values
