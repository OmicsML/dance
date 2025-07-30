# Python Standard Library
import argparse
import copy
import math
import multiprocessing
import pickle
import random
import select
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from re import split
from typing import List, Literal, Optional, Union

# Third-party libraries
import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from igraph import split_join_distance
# PyTorch related
from sklearn import preprocessing
from sklearn.decomposition import NMF
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, NearestNeighbors  # 合并了来自 sklearn.neighbors 的导入
from torch.autograd import Variable
from torch.nn.modules.module import Module  # 注意: 通常直接使用 nn.Module
from torch.nn.parameter import Parameter  # 注意: 通常直接使用 nn.Parameter
# tqdm
from tqdm import tqdm

from dance.data.base import Data
from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.base import BaseRegressionMethod
from dance.modules.spatial.cell_type_deconvo.dstg import GCN
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA, FeatureCellPlaceHolder
from dance.transforms.filter import (
    FilterCellsPlaceHolder,
    FilterGenesCommon,
    FilterGenesPlaceHolder,
    FilterGenesTopK,
    HighlyVariableGenesLogarithmizedByTopGenes,
)
from dance.transforms.misc import Compose, RemoveSplit, SaveRaw, SetConfig
from dance.transforms.normalize import NormalizeTotalLog1P
from dance.typing import LogLevel
from dance.utils.metrics import resolve_score_func
from dance.utils.status import deprecated
from dance.utils.wrappers import add_mod_and_transform


class conGraphConvolutionlayer(Module):

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
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class conGCN(nn.Module):

    def __init__(
        self,
        nfeat,
        nhid,
        common_hid_layers_num,
        fcnn_hid_layers_num,
        dropout,
        nout1,
    ):
        super().__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.common_hid_layers_num = common_hid_layers_num
        self.fcnn_hid_layers_num = fcnn_hid_layers_num
        self.nout1 = nout1
        self.dropout = dropout
        self.training = True

        ## The beginning layer
        self.gc_in_exp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_exp = nn.BatchNorm1d(nhid)
        self.gc_in_sp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_sp = nn.BatchNorm1d(nhid)

        ## common_hid_layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec('self.cgc{}_exp = conGraphConvolutionlayer(nhid, nhid)'.format(i + 1))
                exec('self.bn_node_chid{}_exp = nn.BatchNorm1d(nhid)'.format(i + 1))
                exec('self.cgc{}_sp = conGraphConvolutionlayer(nhid, nhid)'.format(i + 1))
                exec('self.bn_node_chid{}_sp = nn.BatchNorm1d(nhid)'.format(i + 1))

        ## FCNN layers
        self.gc_out11 = nn.Linear(2 * nhid, nhid, bias=True)
        self.bn_out1 = nn.BatchNorm1d(nhid)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec('self.gc_out11{} = nn.Linear(nhid, nhid, bias=True)'.format(i + 1))
                exec('self.bn_out11{} = nn.BatchNorm1d(nhid)'.format(i + 1))
        self.gc_out12 = nn.Linear(nhid, nout1, bias=True)

    def forward(self, x, adjs):

        self.x = x

        ## input layer
        self.x_exp = self.gc_in_exp(self.x, adjs[0])
        self.x_exp = self.bn_node_in_exp(self.x_exp)
        self.x_exp = F.elu(self.x_exp)
        self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        self.x_sp = self.gc_in_sp(self.x, adjs[1])
        self.x_sp = self.bn_node_in_sp(self.x_sp)
        self.x_sp = F.elu(self.x_sp)
        self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## common layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("self.x_exp = self.cgc{}_exp(self.x_exp, adjs[0])".format(i + 1))
                exec("self.x_exp = self.bn_node_chid{}_exp(self.x_exp)".format(i + 1))
                self.x_exp = F.elu(self.x_exp)
                self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
                exec("self.x_sp = self.cgc{}_sp(self.x_sp, adjs[1])".format(i + 1))
                exec("self.x_sp = self.bn_node_chid{}_sp(self.x_sp)".format(i + 1))
                self.x_sp = F.elu(self.x_sp)
                self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## FCNN layers
        self.x1 = torch.cat([self.x_exp, self.x_sp], dim=1)
        self.x1 = self.gc_out11(self.x1)
        self.x1 = self.bn_out1(self.x1)
        self.x1 = F.elu(self.x1)
        self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.x1 = self.gc_out11{}(self.x1)".format(i + 1))
                exec("self.x1 = self.bn_out11{}(self.x1)".format(i + 1))
                self.x1 = F.elu(self.x1)
                self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        self.x1 = self.gc_out12(self.x1)

        gc_list = {}
        gc_list['gc_in_exp'] = self.gc_in_exp
        gc_list['gc_in_sp'] = self.gc_in_sp
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("gc_list['cgc{}_exp'] = self.cgc{}_exp".format(i + 1, i + 1))
                exec("gc_list['cgc{}_sp'] = self.cgc{}_sp".format(i + 1, i + 1))
        gc_list['gc_out11'] = self.gc_out11
        if self.fcnn_hid_layers_num > 0:
            exec("gc_list['gc_out11{}'] =  self.gc_out11{}".format(i + 1, i + 1))
        gc_list['gc_out12'] = self.gc_out12

        return F.log_softmax(self.x1, dim=1), gc_list


def get_idx(train_valid_len, test_len, train_valid_ratio=0.9):
    train_idx = range(int(train_valid_len * train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)
    test_idx = range(test_len)
    return train_idx, valid_idx, test_idx


def conGCN_train(model, train_idx, valid_idx, test_idx, feature, adjs, label, epoch_n, loss_fn, optimizer,
                 scheduler=None, early_stopping_patience=5, clip_grad_max_norm=1, load_test_groundtruth=False,
                 print_epoch_step=1, cpu_num=-1, GCN_device='CPU'):

    if GCN_device == 'CPU':
        device = torch.device("cpu")
        print('Use CPU as device.')
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Use GPU as device.')
        else:
            device = torch.device("cpu")
            print('Use CPU as device.')

    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)

    model = model.to(device)
    adjs = [adj.to(device) for adj in adjs]
    feature = feature.to(device)
    label = label.to(device)

    time_open = time.time()

    best_val = np.inf
    clip = 0
    loss = []
    para_list = []
    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass

        optimizer.zero_grad()
        output1, paras = model(feature.float(), adjs)

        loss_train1 = loss_fn(output1[train_idx], label[train_idx].float())
        loss_val1 = loss_fn(output1[valid_idx], label[valid_idx].float())
        if load_test_groundtruth == True:
            loss_test1 = loss_fn(output1[test_idx], label[test_idx].float())
            loss.append([loss_train1.item(), loss_val1.item(), loss_test1.item()])
        else:
            loss.append([loss_train1.item(), loss_val1.item(), None])

        if epoch % print_epoch_step == 0:
            print("******************************************")
            print("Epoch {}/{}".format(epoch + 1, epoch_n), 'loss_train: {:.4f}'.format(loss_train1.item()),
                  'loss_val: {:.4f}'.format(loss_val1.item()), end='\t')
            if load_test_groundtruth == True:
                print("Test loss= {:.4f}".format(loss_test1.item()), end='\t')
            print('time: {:.4f}s'.format(time.time() - time_open))
        para_list.append(paras.copy())
        for i in paras.keys():
            para_list[-1][i] = copy.deepcopy(para_list[-1][i])

        if early_stopping_patience > 0:
            if torch.round(loss_val1, decimals=4) < best_val:
                best_val = torch.round(loss_val1, decimals=4)
                best_paras = paras.copy()
                best_loss = loss.copy()
                clip = 1
                for i in paras.keys():
                    best_paras[i] = copy.deepcopy(best_paras[i])
            else:
                clip += 1
                if clip == early_stopping_patience:
                    break
        else:
            best_loss = loss.copy()
            best_paras = None

        loss_train1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
        optimizer.step()
        if scheduler != None:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics=loss_val1)

    print("***********************Final Loss***********************")
    print("Epoch {}/{}".format(epoch + 1, epoch_n), 'loss_train: {:.4f}'.format(loss_train1.item()),
          'loss_val: {:.4f}'.format(loss_val1.item()), end='\t')
    if load_test_groundtruth == True:
        print("Test loss= {:.4f}".format(loss_test1.item()), end='\t')
    print('time: {:.4f}s'.format(time.time() - time_open))

    torch.cuda.empty_cache()

    return output1.cpu(), loss, model.cpu()


# def run_STdGCN(

#               ):


def find_mutual_nn(
    data1,
    data2,
    dist_method,
    k1,
    k2,
):
    if dist_method == 'cosine':
        cos_sim1 = cosine_similarity(data1, data2)
        cos_sim2 = cosine_similarity(data2, data1)
        k_index_1 = torch.topk(torch.tensor(cos_sim2), k=k2, dim=1)[1]
        k_index_2 = torch.topk(torch.tensor(cos_sim1), k=k1, dim=1)[1]
    else:
        dist = DistanceMetric.get_metric(dist_method)
        k_index_1 = KDTree(data1, metric=dist).query(data2, k=k2, return_distance=False)
        k_index_2 = KDTree(data2, metric=dist).query(data1, k=k1, return_distance=False)
    mutual_1 = []
    mutual_2 = []
    mutual = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
                mutual.append([index_1, index_2])
    return mutual


def inter_adj(
    ST_integration,
    find_neighbor_method='MNN',
    dist_method='euclidean',
    corr_dist_neighbors=20,
):

    if find_neighbor_method == 'KNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        if dist_method == 'cosine':
            cos_sim = cosine_similarity(data1, data2)
            k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
        else:
            dist = DistanceMetric.get_metric(dist_method)
            k_index = KDTree(data2, metric=dist).query(data1, k=corr_dist_neighbors, return_distance=False)
        A_exp = np.zeros((ST_integration.shape[0], ST_integration.shape[0]), dtype=float)
        for i in range(k_index.shape[0]):
            for j in k_index[i]:
                A_exp[i, j + real_num] = 1
                A_exp[j + real_num, i] = 1
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)

    elif find_neighbor_method == 'MNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        mut = find_mutual_nn(data2, data1, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
        mut = pd.DataFrame(mut, columns=['pseudo', 'real'])
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        A_exp = np.zeros((real_num + pseudo_num, real_num + pseudo_num), dtype=float)
        for i in mut.index:
            A_exp[mut.loc[i, 'real'], mut.loc[i, 'pseudo'] + real_num] = 1
            A_exp[mut.loc[i, 'pseudo'] + real_num, mut.loc[i, 'real']] = 1
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)

    return A_exp


def intra_dist_adj(ST_exp, link_method='soft', space_dist_neighbors=27, space_dist_threshold=None):

    knn = NearestNeighbors(n_neighbors=space_dist_neighbors, metric='minkowski')

    # knn.fit(ST_exp.obs[['coor_X', 'coor_Y']])
    knn.fit(ST_exp.obsm['spatial'][['x', 'y']])
    dist, ind = knn.kneighbors()

    if link_method == 'hard':
        A_space = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                if space_dist_threshold != None:
                    if dist[i, j] < space_dist_threshold:
                        A_space[i, ind[i, j]] = 1
                        A_space[ind[i, j], i] = 1
                else:
                    A_space[i, ind[i, j]] = 1
                    A_space[ind[i, j], i] = 1
        A_space = pd.DataFrame(A_space, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    else:
        A_space = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                if space_dist_threshold != None:
                    if dist[i, j] < space_dist_threshold:
                        A_space[i, ind[i, j]] = 1 / dist[i, j]
                        A_space[ind[i, j], i] = 1 / dist[i, j]
                else:
                    A_space[i, ind[i, j]] = 1 / dist[i, j]
                    A_space[ind[i, j], i] = 1 / dist[i, j]
        A_space = pd.DataFrame(A_space, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)

    return A_space


def intra_exp_adj(
    adata,
    find_neighbor_method='KNN',
    dist_method='euclidean',
    PCA_dimensionality_reduction=True,
    dim=50,
    corr_dist_neighbors=10,
    channel: Optional[str] = "feature.cell",
    channel_type: str = "obsm",
):

    ST_exp = adata.copy()

    # sc.pp.scale(ST_exp, max_value=None, zero_center=True)
    if PCA_dimensionality_reduction == True:
        sc.tl.pca(ST_exp, n_comps=dim, svd_solver='arpack', random_state=None)
        input_data = ST_exp.obsm['X_pca']
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors,
                                                                return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1
                        A_exp[j, i] = 1
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors,
                                 k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    else:
        # sc.pp.scale(ST_exp, max_value=None, zero_center=True)
        if channel_type is None:
            input_data = ST_exp.X
        else:
            input_data = ST_exp.obsm[channel]
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors,
                                                                return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1
                        A_exp[j, i] = 1
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors,
                                 k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)

    return A_exp


def A_intra_transfer(data, data_type, real_num, pseudo_num):

    adj = np.zeros((real_num + pseudo_num, real_num + pseudo_num), dtype=float)
    if data_type == 'real':
        adj[:real_num, :real_num] = data
    elif data_type == 'pseudo':
        adj[real_num:, real_num:] = data

    return adj


def adj_normalize(mx, symmetry=True):

    mx = sp.csr_matrix(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    if symmetry == True:
        r_mat_inv = sp.diags(np.sqrt(r_inv))
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    else:
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

    return mx.todense()


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class autoencoder(nn.Module):

    def __init__(self, x_size, hidden_size, embedding_size, p_drop=0):
        super().__init__()

        self.encoder = nn.Sequential(full_block(x_size, hidden_size, p_drop),
                                     full_block(hidden_size, embedding_size, p_drop))

        self.decoder = nn.Sequential(full_block(embedding_size, hidden_size, p_drop),
                                     full_block(hidden_size, x_size, p_drop))

    def forward(self, x):

        en = self.encoder(x)
        de = self.decoder(en)

        return en, de, [self.encoder, self.decoder]


def auto_train(model, epoch_n, loss_fn, optimizer, data, cpu_num=-1, device='GPU'):

    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)

    if device == 'GPU':
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()

    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass

        train_cost = 0

        optimizer.zero_grad()
        en, de, _ = model(data)

        loss = loss_fn(de, data)

        loss.backward()
        optimizer.step()

    torch.cuda.empty_cache()

    return en.cpu()


@register_preprocessor("normalize")
@add_mod_and_transform
@deprecated(msg="will be replaced by builtin bypass mechanism in pipeline")
class STPreprocessTransform(BaseTransform):
    """Used as a placeholder to skip the process."""

    def __init__(self, normalize=True, log=True, highly_variable_genes=False, regress_out=False, scale=False,
                 scale_max_value=None, scale_zero_center=True, hvg_min_mean=0.0125, hvg_max_mean=3, hvg_min_disp=0.5,
                 highly_variable_gene_num=None, split="ref", **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.log = log
        self.highly_variable_genes = highly_variable_genes
        self.regress_out = regress_out
        self.scale = scale
        self.scale_max_value = scale_max_value
        self.scale_zero_center = scale_zero_center
        self.hvg_min_mean = hvg_min_mean
        self.hvg_max_mean = hvg_max_mean
        self.hvg_min_disp = hvg_min_disp
        self.highly_variable_gene_num = highly_variable_gene_num
        self.split = split

    def __call__(self, data: Data) -> Data:
        ST_exp = data.get_split_data(split_name=self.split)
        adata = ST_exp
        if self.normalize == True:
            sc.pp.normalize_total(adata, target_sum=1e4)

        if self.log == True:
            sc.pp.log1p(adata)

        adata.layers['scale.data'] = adata.X.copy()

        if self.highly_variable_genes == True:
            sc.pp.highly_variable_genes(
                adata,
                min_mean=self.hvg_min_mean,
                max_mean=self.hvg_max_mean,
                min_disp=self.hvg_min_disp,
                n_top_genes=self.highly_variable_gene_num,
            )
            adata = adata[:, adata.var.highly_variable]

        if self.regress_out == True:
            mito_genes = adata.var_names.str.startswith('MT-')
            adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
            sc.pp.filter_cells(adata, min_counts=0)
            sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

        if self.scale == True:
            sc.pp.scale(adata, max_value=self.scale_max_value, zero_center=self.scale_zero_center)
        selected_genes = adata.uns['gene_list']
        real_selected_genes = list(set(data.data.var_names).intersection(set(selected_genes)))
        # data.data = adata[:, selected_genes]
        data.data._inplace_subset_var(real_selected_genes)


def generate_a_spot_optimized(
    # Pass pre-processed data to avoid repeated computations and large object passing
    cells_by_type_indices: dict,  # {cell_type_name: [list of integer indices for that type]}
    all_unique_cell_types: list,  # List of all unique cell type names
    cell_type_to_original_name_map: dict,  # if obs['cellType'] contains integers, map them back to names for consistency
    min_cell_number_in_spot: int,
    max_cell_number_in_spot: int,
    max_cell_types_in_spot: int,
    generation_method: str,
    # For 'cell' method, we'd need total_cell_count or list of all cell indices
    all_cell_indices: Optional[list] = None,
):

    picked_cell_indices = []  # Store integer indices relative to the original sc_exp
    picked_cell_type_names = []  # Store the cell type name for each picked cell

    if generation_method == 'cell':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        # Ensure all_cell_indices is provided for 'cell' method
        if all_cell_indices is None:
            raise ValueError("all_cell_indices must be provided for 'cell' generation method.")

        # random.choices can pick the same cell multiple times, which matches original logic
        picked_cell_indices = random.choices(all_cell_indices, k=cell_num)
        # We need to get the types for these picked cells. This requires sc_exp.obs['cellType']
        # This part is tricky without passing sc_exp.obs['cellType'] or a mapping.
        # For simplicity in this optimized worker, we'll assume 'cell' method might need rethinking
        # or the main process will handle type lookups based on returned indices.
        # For now, let's return empty type names for 'cell' method and let caller handle it.
        picked_cell_type_names = [None] * cell_num  # Placeholder

    elif generation_method == 'celltype':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)

        # Ensure max_cell_types_in_spot is not greater than available unique types
        num_types_to_sample = min(max_cell_types_in_spot, len(all_unique_cell_types))
        if num_types_to_sample <= 0:  # If no cell types available
            return [], []  # Return empty if no types to sample from

        actual_cell_type_num_in_spot = random.randint(1, num_types_to_sample)

        # Efficiently sample unique cell types
        selected_cell_types_for_spot = random.sample(all_unique_cell_types, k=actual_cell_type_num_in_spot)

        for _ in range(cell_num):
            # Pick a cell type from the selected types for this spot (with replacement)
            chosen_cell_type_name = random.choice(selected_cell_types_for_spot)

            # Pick a cell index from that chosen cell type (with replacement)
            # Ensure the chosen cell type has cells available
            if cells_by_type_indices[chosen_cell_type_name]:
                cell_idx = random.choice(cells_by_type_indices[chosen_cell_type_name])
                picked_cell_indices.append(cell_idx)
                picked_cell_type_names.append(chosen_cell_type_name)
            else:
                # This case should ideally not happen if all_unique_cell_types is derived from
                # keys of cells_by_type_indices where lists are non-empty.
                # If it can happen, decide on a fallback (e.g., skip, or try another type)
                # For now, we'll just potentially have fewer than cell_num cells if a type is empty.
                pass
    else:
        # This part of the function won't be reached if called from pseudoSpotGen
        # as the check is done there. But good for direct use.
        raise ValueError('generation_method should be "cell" or "celltype"')

    return picked_cell_indices, picked_cell_type_names


import numpy as np

# from scipy.sparse import csr_matrix # 如果 sc_exp.X 可能不是csr但仍是稀疏的


def process_single_spot(i, spot_data_item, sc_exp, word_to_idx_celltype, generation_method):
    """Processes a single spot's data.

    Returns the index i, the expression sum array, and a dictionary of type_idx counts
    for this spot.

    """
    spot_cell_indices, spot_cell_type_names = spot_data_item

    if not spot_cell_indices:
        return i, None, {}  # Return None for expression, empty dict for counts

    spot_expression_sum = None
    # Sum expression for the spot
    if len(spot_cell_indices) > 0:
        # Ensure spot_cell_indices are valid if using sparse X
        valid_indices = [idx for idx in spot_cell_indices if idx < sc_exp.shape[0]]

        if not valid_indices:  # If all indices were invalid
            return i, None, {}

        if hasattr(sc_exp.X, "tocsr"):  # Check if sparse (issparse is more general)
            # It's generally better to check `scipy.sparse.issparse(sc_exp.X)`
            # .A1 flattens to 1D array. If sc_exp.X is already 1D per cell, .A1 might not be needed after sum.
            spot_expression_sum = sc_exp.X[valid_indices, :].sum(axis=0)
            if hasattr(spot_expression_sum, 'A1'):  # If it's a matrix, convert to array
                spot_expression_sum = spot_expression_sum.A1
        else:  # Dense
            spot_expression_sum = sc_exp.X[valid_indices, :].sum(axis=0)

    # Calculate fractions for this spot
    spot_fraction_counts = {}  # Store counts for type_idx for *this* spot
    for cell_type_name in spot_cell_type_names:
        if cell_type_name is None and generation_method == 'cell':
            # This logic needs to be fully implemented if 'cell' method types aren't pre-fetched.
            # For example, if you need to look up sc_exp.obs['cellType'] based on an index in spot_cell_indices:
            # You'd need to iterate through spot_cell_indices as well, not just spot_cell_type_names,
            # or ensure spot_cell_type_names is correctly populated *before* this stage.
            # For now, we'll assume spot_cell_type_names is what we should use.
            # print(f"Warning (Spot {i}): 'cell' method with None cell_type_name encountered. Requires specific handling to fetch type from sc_exp.obs.")
            pass

        if cell_type_name is not None:
            type_idx = word_to_idx_celltype.get(cell_type_name)
            if type_idx is not None:
                spot_fraction_counts[type_idx] = spot_fraction_counts.get(type_idx, 0) + 1
            else:
                # Be careful with print statements in threads, they can be messy.
                # Consider logging or collecting warnings.
                # print(f"Warning (Spot {i}): Cell type '{cell_type_name}' not found in word_to_idx_celltype map.")
                pass  # Or collect these warnings

    return i, spot_expression_sum, spot_fraction_counts


@register_preprocessor("pseudobulk")
class pseudoSpotGen(BaseTransform):

    def __init__(self, spot_num, min_cell_number_in_spot, max_cell_number_in_spot, max_cell_types_in_spot,
                 generation_method, n_jobs=-1, in_split_name: str = "ref", out_split_name: Optional[str] = "pseudo",
                 **kwargs):
        super().__init__(**kwargs)
        self.spot_num = spot_num
        self.min_cell_number_in_spot = min_cell_number_in_spot
        self.max_cell_number_in_spot = max_cell_number_in_spot
        self.max_cell_types_in_spot = max_cell_types_in_spot
        self.generation_method = generation_method
        self.n_jobs = n_jobs
        self.in_split_name = in_split_name
        self.out_split_name = out_split_name
        if self.generation_method not in ['cell', 'celltype']:
            raise ValueError('generation_method should be "cell" or "celltype"')

    def __call__(self, data: Data) -> Data:
        sc_exp = data.get_split_data(self.in_split_name)

        # Ensure 'cellType' and 'cell_type_idx' exist
        if 'cellType' not in sc_exp.obs.columns:
            raise ValueError("'cellType' column not found in sc_exp.obs")
        if 'cell_type_idx' not in sc_exp.obs.columns:
            # If 'cell_type_idx' is missing, try to create it
            print("Warning: 'cell_type_idx' not found in sc_exp.obs. Creating it from 'cellType'.")
            sc_exp.obs['cellType'] = sc_exp.obs['cellType'].astype('category')
            sc_exp.obs['cell_type_idx'] = sc_exp.obs['cellType'].cat.codes
            # Also update/create idx_to_word_celltype if it's tied to this
            data.data.uns['idx_to_word_celltype'] = {
                i: cat
                for i, cat in enumerate(sc_exp.obs['cellType'].cat.categories)
            }

        idx_to_word_celltype = data.data.uns['idx_to_word_celltype']
        word_to_idx_celltype = {v: k for k, v in idx_to_word_celltype.items()}

        num_distinct_cell_types_in_sc = len(sc_exp.obs['cellType'].unique())

        # Pre-process data for workers to minimize data transfer and redundant computation
        all_unique_cell_type_names = list(sc_exp.obs['cellType'].unique())

        # Create a mapping from cell type name to a list of *integer indices* of cells of that type
        # Using .to_numpy() for sc_exp.obs['cellType'] can be faster for large datasets
        cell_types_array = sc_exp.obs['cellType'].to_numpy()
        cells_by_type_indices = {}
        for ct_name in all_unique_cell_type_names:
            # np.where returns a tuple of arrays, we need the first array
            indices = np.where(cell_types_array == ct_name)[0].tolist()
            cells_by_type_indices[ct_name] = indices

        all_sc_exp_cell_indices = None
        if self.generation_method == 'cell':
            all_sc_exp_cell_indices = list(range(sc_exp.n_obs))

        cores = multiprocessing.cpu_count()
        n_jobs = self.n_jobs
        if n_jobs == -1:
            n_jobs = cores
        else:
            n_jobs = min(n_jobs, cores)

        pool = multiprocessing.Pool(processes=n_jobs)

        args_list = [
            (
                cells_by_type_indices,
                all_unique_cell_type_names,
                {},  # cell_type_to_original_name_map (not strictly needed if cellType is already names)
                self.min_cell_number_in_spot,
                self.max_cell_number_in_spot,
                self.max_cell_types_in_spot,
                self.generation_method,
                all_sc_exp_cell_indices)  # Pass all cell indices for 'cell' method
            for _ in range(self.spot_num)
        ]

        # generated_spot_data will be a list of tuples: (picked_cell_indices, picked_cell_type_names)
        generated_spot_data = []
        # Use imap_unordered for potentially better memory usage with large number of spots,
        # and tqdm for progress bar. Or stick to starmap if order matters and for simplicity.
        # generated_spot_data = list(tqdm(pool.imap_unordered(lambda p: generate_a_spot_optimized(*p), args_list),
        #                                 total=self.spot_num, desc='Generating pseudo-spots (data)'))
        generated_spot_data = pool.starmap(generate_a_spot_optimized,
                                           tqdm(args_list, desc='Generating pseudo-spots (data)'))

        pool.close()
        pool.join()

        pseudo_spots_table_X = np.zeros((self.spot_num, sc_exp.shape[1]),
                                        dtype=np.float32)  # Use float32 to save memory if precision allows
        pseudo_fraction_table_counts = np.zeros((self.spot_num, num_distinct_cell_types_in_sc), dtype=int)

        # To pass large objects like sc_exp efficiently, ensure they are shared correctly.
        # For threads, direct access to self.sc_exp is fine.
        # If sc_exp.X is extremely large and you were using multiprocessing,
        # you might investigate shared memory, but for threading this is not an issue.

        # --- Part 1: Expression Summation Vectorization ---
        print("Preparing data for expression summation...")
        spot_indices_for_S_matrix = []
        cell_indices_for_S_matrix = []

        valid_spot_indices_for_expr = []  # Keep track of spots that are not empty for expression

        for i in range(self.spot_num):
            spot_cell_indices, _ = generated_spot_data[i]
            if not spot_cell_indices:
                continue

            # Filter for valid cell indices (within sc_exp.shape[0])
            valid_indices_in_spot = [idx for idx in spot_cell_indices if idx < sc_exp.shape[0]]

            if not valid_indices_in_spot:
                continue

            valid_spot_indices_for_expr.append(i)
            spot_indices_for_S_matrix.extend([i] * len(valid_indices_in_spot))
            cell_indices_for_S_matrix.extend(valid_indices_in_spot)

        if spot_indices_for_S_matrix:  # Only proceed if there's data
            data_for_S_matrix = np.ones(len(cell_indices_for_S_matrix), dtype=int)

            # Ensure S_matrix has self.spot_num rows, even if some are empty
            S_matrix = sp.csr_matrix((data_for_S_matrix, (spot_indices_for_S_matrix, cell_indices_for_S_matrix)),
                                     shape=(self.spot_num, sc_exp.shape[0]))

            print("Calculating expression sums via sparse matrix multiplication...")
            # This is the core vectorized operation for expression
            summed_expressions = S_matrix @ sc_exp.X

            # If summed_expressions is sparse, convert to dense for assignment
            # or assign row by row if memory is an issue for full dense conversion
            if sp.issparse(summed_expressions):
                # pseudo_spots_table_X[valid_spot_indices_for_expr] = summed_expressions[valid_spot_indices_for_expr].toarray() # This might not work directly if S_matrix had all spot_num rows
                # A safer way if S_matrix was built with self.spot_num rows:
                for i_idx, original_spot_idx in enumerate(
                        np.unique(spot_indices_for_S_matrix)):  # Iterate over spots that actually had cells
                    # Find rows in summed_expressions that correspond to original_spot_idx
                    # This can be tricky if S_matrix was built with fewer rows and then padded.
                    # Assuming S_matrix was built with shape (self.spot_num, sc_exp.shape[0])
                    if summed_expressions[original_spot_idx].nnz > 0:  # check if the row is not all zero
                        pseudo_spots_table_X[original_spot_idx] = summed_expressions[original_spot_idx].toarray().ravel(
                        )
                    # else it remains zero, which is correct for empty or invalid spots
            else:  # If sc_exp.X was dense, summed_expressions is dense
                # pseudo_spots_table_X[valid_spot_indices_for_expr] = summed_expressions[valid_spot_indices_for_expr]
                # Simpler:
                pseudo_spots_table_X = summed_expressions  # if S_matrix already has self.spot_num rows
        else:
            print("No valid cells found across all spots for expression summation.")

        # --- Part 2: Fraction Calculation Vectorization ---
        print("Preparing data for fraction calculation...")
        spot_indices_flat_for_fractions = []
        cell_type_indices_flat_for_fractions = []

        # Handling for generation_method == 'cell' and None cell_type_name
        # This part needs careful consideration if types are not pre-fetched.
        # For now, assuming spot_cell_type_names is mostly usable.
        # If lookups are needed, this pre-processing step might need its own loop,
        # potentially parallelized if it's slow.

        for i in range(self.spot_num):
            spot_cell_indices, spot_cell_type_names = generated_spot_data[i]

            if not spot_cell_indices:  # or not spot_cell_type_names, depending on logic
                continue

            processed_cell_type_names_for_spot = []
            if self.generation_method == 'cell':
                # Example: If spot_cell_type_names can be None and needs lookup
                # This is a placeholder for the complex logic mentioned in your original code.
                # You might need to iterate 'spot_cell_indices' and use 'sc_exp.obs'
                # For simplicity here, we assume spot_cell_type_names is populated.
                # If a cell type is None, it will be skipped by .get() later.
                for j, cell_type_name in enumerate(spot_cell_type_names):
                    if cell_type_name is None:
                        # Lookup logic: e.g. cell_type_name = self.sc_exp.obs['cellType'].iloc[spot_cell_indices[j]]
                        # This lookup can be slow if done cell by cell.
                        # print(f"Warning (Spot {i}): 'cell' method with None cell_type_name. Implement lookup.")
                        pass  # Placeholder for actual lookup
                    if cell_type_name is not None:
                        processed_cell_type_names_for_spot.append(cell_type_name)
            else:  # 'celltype' method or types are already good
                processed_cell_type_names_for_spot = [ctn for ctn in spot_cell_type_names if ctn is not None]

            current_spot_type_indices = []
            for cell_type_name in processed_cell_type_names_for_spot:
                type_idx = word_to_idx_celltype.get(cell_type_name)
                if type_idx is not None:
                    current_spot_type_indices.append(type_idx)
                else:
                    print(f"Warning: Cell type '{cell_type_name}' not found in word_to_idx_celltype map for spot {i}.")

            if current_spot_type_indices:
                spot_indices_flat_for_fractions.extend([i] * len(current_spot_type_indices))
                cell_type_indices_flat_for_fractions.extend(current_spot_type_indices)

        if spot_indices_flat_for_fractions:
            print("Calculating fractions using np.add.at...")
            # Convert to numpy arrays for np.add.at
            spot_indices_np = np.array(spot_indices_flat_for_fractions, dtype=int)
            cell_type_indices_np = np.array(cell_type_indices_flat_for_fractions, dtype=int)

            # Ensure indices are within bounds for pseudo_fraction_table_counts
            valid_fraction_mask = (spot_indices_np < pseudo_fraction_table_counts.shape[0]) & \
                                  (cell_type_indices_np < pseudo_fraction_table_counts.shape[1])

            if not np.all(valid_fraction_mask):
                print("Warning: Some indices for fraction calculation are out of bounds and will be skipped.")
                spot_indices_np = spot_indices_np[valid_fraction_mask]
                cell_type_indices_np = cell_type_indices_np[valid_fraction_mask]

            if spot_indices_np.size > 0:
                np.add.at(pseudo_fraction_table_counts, (spot_indices_np, cell_type_indices_np), 1)
        else:
            print("No valid cell types found across all spots for fraction calculation.")

        pseudo_spots_adata = ad.AnnData(X=pseudo_spots_table_X, var=sc_exp.var.copy())  # Keep var names
        pseudo_spots_adata.obs.index = [f"pseudo_spot_{j}" for j in range(self.spot_num)]  # Meaningful spot names

        # Create DataFrame for fractions
        type_list_ordered = [idx_to_word_celltype[i] for i in range(num_distinct_cell_types_in_sc)]
        pseudo_fraction_df = pd.DataFrame(pseudo_fraction_table_counts, columns=type_list_ordered,
                                          index=pseudo_spots_adata.obs.index)

        pseudo_fraction_df['cell_num'] = pseudo_fraction_df.sum(axis=1)

        for col in type_list_ordered:  # Iterate only over actual type columns
            # Avoid division by zero if 'cell_num' is 0
            pseudo_fraction_df[col] = np.where(pseudo_fraction_df['cell_num'] > 0,
                                               pseudo_fraction_df[col] / pseudo_fraction_df['cell_num'], 0)

        # Join fractions into the new AnnData's obs
        pseudo_spots_adata.obs = pseudo_spots_adata.obs.join(pseudo_fraction_df)

        data.append(Data(pseudo_spots_adata), join="outer", mode="new_split", new_split_name=self.out_split_name)
        return data


# #remove split ref
# def data_integration(real, pseudo, batch_removal_method="combat", dimensionality_reduction_method='PCA', dim=50,
#                      scale=True, autoencoder_epoches=2000, autoencoder_LR=1e-3, autoencoder_drop=0, cpu_num=-1,
#                      AE_device='GPU'):

#     if batch_removal_method == 'mnn':
#         mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True,
#                                          var_subset=None)
#         adata = mnn[0]
#         if dimensionality_reduction_method == 'PCA':
#             if scale == True:
#                 sc.pp.scale(adata, max_value=None, zero_center=True)
#             sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
#             table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == 'autoencoder':
#             data = torch.tensor(adata.X)
#             x_size = data.shape[1]
#             latent_size = dim
#             hidden_size = int((x_size + latent_size) / 2)
#             nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
#                                p_drop=autoencoder_drop)
#             optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
#             loss_ae = nn.MSELoss(reduction='mean')
#             embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
#                                    data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
#             if scale == True:
#                 embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
#             table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == 'nmf':
#             nmf = NMF(n_components=dim).fit_transform(adata.X)
#             if scale == True:
#                 nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
#             table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == None:
#             if scale == True:
#                 sc.pp.scale(adata, max_value=None, zero_center=True)
#             table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
#         table = table.iloc[pseudo.shape[0]:, :].append(table.iloc[:pseudo.shape[0], :])
#         table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

#     elif batch_removal_method == 'scanorama':
#         import scanorama
#         scanorama.integrate_scanpy([real, pseudo], dimred=dim)
#         table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
#         table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
#         table = table1.append(table2)
#         table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

#     elif batch_removal_method == 'combat':
#         aaa = real.copy()
#         aaa.obs = pd.DataFrame(index=aaa.obs.index)
#         bbb = pseudo.copy()
#         bbb.obs = pd.DataFrame(index=bbb.obs.index)
#         adata = aaa.concatenate(bbb, batch_key='real_pseudo')
#         sc.pp.combat(adata, key='real_pseudo')
#         if dimensionality_reduction_method == 'PCA':
#             if scale == True:
#                 sc.pp.scale(adata, max_value=None, zero_center=True)
#             sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
#             table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == 'autoencoder':
#             data = torch.tensor(adata.X)
#             x_size = data.shape[1]
#             latent_size = dim
#             hidden_size = int((x_size + latent_size) / 2)
#             nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
#                                p_drop=autoencoder_drop)
#             optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
#             loss_ae = nn.MSELoss(reduction='mean')
#             embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
#                                    data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
#             if scale == True:
#                 embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
#             table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == 'nmf':
#             nmf = NMF(n_components=dim).fit_transform(adata.X)
#             if scale == True:
#                 nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
#             table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == None:
#             if scale == True:
#                 sc.pp.scale(adata, max_value=None, zero_center=True)
#             table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
#         table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

#     else:
#         aaa = real.copy()
#         aaa.obs = pd.DataFrame(index=aaa.obs.index)
#         bbb = pseudo.copy()
#         bbb.obs = pd.DataFrame(index=bbb.obs.index)
#         adata = aaa.concatenate(bbb, batch_key='real_pseudo')
#         if dimensionality_reduction_method == 'PCA':
#             if scale == True:
#                 sc.pp.scale(adata, max_value=None, zero_center=True)
#             sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
#             table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == 'autoencoder':
#             data = torch.tensor(adata.X)
#             x_size = data.shape[1]
#             latent_size = dim
#             hidden_size = int((x_size + latent_size) / 2)
#             nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
#                                p_drop=autoencoder_drop)
#             optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
#             loss_ae = nn.MSELoss(reduction='mean')
#             embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
#                                    data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
#             if scale == True:
#                 embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
#             table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == 'nmf':
#             nmf = NMF(n_components=dim).fit_transform(adata.X)
#             if scale == True:
#                 nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
#             table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
#         elif dimensionality_reduction_method == None:
#             if scale == True:
#                 sc.pp.scale(adata, max_value=None, zero_center=False)
#             table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
#         table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

#     table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist() + pseudo.obs['cell_num'].values.tolist())
#     table.insert(2, 'cell_type_num',
#                  real.obs['cell_type_num'].values.tolist() + pseudo.obs['cell_type_num'].values.tolist())

#     return table

# !/usr/bin/env python
# coding: utf-8

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())


@register_preprocessor("misc")
class CelltypeTransform(BaseTransform):
    """Cell topic profile."""

    _DISPLAY_ATTRS = ("ct_select", "ct_key", "split_name", "method")

    def __init__(
        self,
        *,
        ct_select: Union[Literal["auto"], List[str]] = "auto",
        ct_key: str = "cellType",
        batch_key: Optional[str] = None,
        split_name: Optional[str] = "ref",
        channel: Optional[str] = None,
        channel_type: str = "X",
        method: Literal["median", "mean"] = "median",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ct_select = ct_select
        self.ct_key = ct_key
        self.split_name = split_name

        self.channel = channel
        self.channel_type = channel_type
        self.method = method

    def __call__(self, data):
        x = data.get_feature(split_name=self.split_name, channel=self.channel, channel_type=self.channel_type,
                             return_type="numpy")
        annot = data.get_feature(split_name=self.split_name, channel=self.ct_key, channel_type="obs",
                                 return_type="numpy")

        # sc_adata = ad.AnnData(X=x)
        # sc_adata.obs['cell_type'] = annot
        # data.
        # cell_type_num = len(annot.unique())
        # print(annot)
        cell_types = np.unique(annot)

        word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
        idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}
        split_idx = data.get_split_idx(self.split_name)
        celltype_idx = [word_to_idx_celltype[w] for w in annot]
        # Initialize the column first if it doesn't exist
        if 'cell_type_idx' not in data.data.obs.columns:
            data.data.obs['cell_type_idx'] = pd.Series(index=data.data.obs_names, dtype=int)  # or float, or object

        # This might still give a SettingWithCopyWarning from pandas itself
        # if adata.obs itself is somehow a view, but less likely for adata.obs.
        data.data.obs['cell_type_idx'].iloc[split_idx] = celltype_idx

        # sc_adata.obs['cell_type'].value_counts()
        # data.data=sc_adata
        data.data.uns['idx_to_word_celltype'] = idx_to_word_celltype
        data.data.uns['word_to_idx_celltype'] = word_to_idx_celltype
        data.data.uns['cell_types_list'] = cell_types


class stdGCNMarkGenes(BaseTransform):

    def __init__(self, preprocess=True, highly_variable_genes=True, regress_out=False, scale=False, PCA_components=50,
                 marker_gene_method='wilcoxon', filter_wilcoxon_marker_genes=True, top_gene_per_type=20,
                 pvals_adj_threshold=0.10, log_fold_change_threshold=1, min_within_group_fraction_threshold=0.7,
                 max_between_group_fraction_threshold=0.3, split="ref", **kwargs):
        super().__init__(**kwargs)
        self.preprocess = preprocess
        self.highly_variable_genes = highly_variable_genes
        self.regress_out = regress_out
        self.scale = scale
        self.PCA_components = PCA_components
        self.marker_gene_method = marker_gene_method
        self.filter_wilcoxon_marker_genes = filter_wilcoxon_marker_genes
        self.top_gene_per_type = top_gene_per_type
        self.pvals_adj_threshold = pvals_adj_threshold
        self.log_fold_change_threshold = log_fold_change_threshold
        self.min_within_group_fraction_threshold = min_within_group_fraction_threshold
        self.max_between_group_fraction_threshold = max_between_group_fraction_threshold
        self.split = split

    def __call__(self, data: Data) -> Data:
        sc_exp = data.get_split_data(self.split)
        # if preprocess == True:
        #     sc_adata_marker_gene = ST_preprocess(
        #         sc_exp.copy(),
        #         normalize=True,
        #         log=True,
        #         highly_variable_genes=highly_variable_genes,
        #         regress_out=regress_out,
        #         scale=scale,
        #     )
        # else:
        #     sc_adata_marker_gene = sc_exp.copy()
        sc_adata_marker_gene = sc_exp
        sc.tl.pca(sc_adata_marker_gene, n_comps=self.PCA_components, svd_solver='arpack', random_state=None)

        # layer = 'scale.data'
        layer = None
        sc.tl.rank_genes_groups(sc_adata_marker_gene, 'cellType', layer=layer, use_raw=False, pts=True,
                                method=self.marker_gene_method, corr_method='benjamini-hochberg',
                                key_added=self.marker_gene_method)

        if self.marker_gene_method == 'wilcoxon':
            if self.filter_wilcoxon_marker_genes == True:
                gene_dict = {}
                gene_list = []
                for name in sc_adata_marker_gene.obs['cellType'].unique():
                    data = sc.get.rank_genes_groups_df(sc_adata_marker_gene, group=name,
                                                       key=self.marker_gene_method).sort_values('pvals_adj')
                    if self.pvals_adj_threshold != None:
                        data = data[data['pvals_adj'] < self.pvals_adj_threshold]
                    if self.log_fold_change_threshold != None:
                        data = data[data['logfoldchanges'] >= self.log_fold_change_threshold]
                    if self.min_within_group_fraction_threshold != None:
                        data = data[data['pct_nz_group'] >= self.min_within_group_fraction_threshold]
                    if self.max_between_group_fraction_threshold != None:
                        data = data[data['pct_nz_reference'] < self.max_between_group_fraction_threshold]
                    gene_dict[name] = data['names'].values[:self.top_gene_per_type].tolist()
                    gene_list = gene_list + data['names'].values[:self.top_gene_per_type].tolist()
                    gene_list = list(set(gene_list))
            else:
                gene_table = pd.DataFrame(
                    sc_adata_marker_gene.uns[self.marker_gene_method]['names'][:self.top_gene_per_type])
                gene_dict = {}
                for i in gene_table.columns:
                    gene_dict[i] = gene_table[i].values.tolist()
                gene_list = list({item for sublist in gene_table.values.tolist() for item in sublist})
        elif self.marker_gene_method == 'logreg':
            gene_table = pd.DataFrame(
                sc_adata_marker_gene.uns[self.marker_gene_method]['names'][:self.top_gene_per_type])
            gene_dict = {}
            for i in gene_table.columns:
                gene_dict[i] = gene_table[i].values.tolist()
            gene_list = list({item for sublist in gene_table.values.tolist() for item in sublist})
        else:
            print("marker_gene_method should be 'logreg' or 'wilcoxon'")
        data.data.uns['gene_list'] = gene_list
        data.data.uns['gene_dict'] = gene_dict


@register_preprocessor("misc")
class updateAnndataObsTransform(BaseTransform):

    def __init__(self, split, **kwargs):
        self.split = split
        super().__init__(**kwargs)

    def __call__(self, data: Data):
        # target_adata: ad.AnnData, source_adata: Optional[ad.AnnData] = None,
        #                 cell_types_list: Optional[List[str]] = None
        cell_types_list = data.data.uns['cell_types_list']
        target_adata = data.get_split_data(self.split)
        source_adata = target_adata.raw
        if target_adata is None:
            raise ValueError("target_adata cannot be None.")

        num_obs_target = target_adata.shape[0]
        try:
            if source_adata is not None and 'cell_num' in source_adata.obs:
                source_cell_num_data = source_adata.obs['cell_num']
                try:
                    target_adata.obs.insert(0, 'cell_num', source_cell_num_data)
                    print("Info: 'cell_num' inserted from source_adata.")
                except (ValueError, Exception) as e_insert:  # ValueError if column exists
                    target_adata.obs['cell_num'] = source_cell_num_data
                    print(f"Info: 'cell_num' assigned from source_adata (insertion failed: {e_insert}).")
            else:
                raise KeyError("'cell_num' not in source_adata or source_adata is None.")
        except (KeyError, AttributeError) as e_source:  # AttributeError if source_adata is None
            print(f"Warning: Could not get 'cell_num' from source_adata ({e_source}). Initializing with zeros.")
            default_cell_num = [0] * num_obs_target
            try:
                target_adata.obs.insert(0, 'cell_num', default_cell_num)
                print("Info: 'cell_num' inserted with default zeros.")
            except (ValueError, Exception) as e_insert_default:
                target_adata.obs['cell_num'] = default_cell_num
                print(f"Info: 'cell_num' assigned with default zeros (insertion failed: {e_insert_default}).")
        if cell_types_list is not None:
            for ct_col in cell_types_list:
                try:
                    if source_adata is not None and ct_col in source_adata.obs:
                        target_adata.obs[ct_col] = source_adata.obs[ct_col]
                        print(f"Info: Column '{ct_col}' copied from source_adata.")
                    else:
                        raise KeyError(f"'{ct_col}' not in source_adata or source_adata is None.")
                except (KeyError, AttributeError) as e_source_ct:
                    print(
                        f"Warning: Could not get '{ct_col}' from source_adata ({e_source_ct}). Initializing with zeros."
                    )
                    target_adata.obs[ct_col] = [0] * num_obs_target
                except Exception as e_assign:
                    print(f"Error: Failed to assign '{ct_col}'. Initializing with zeros. Details: {e_assign}")
                    target_adata.obs[ct_col] = [0] * num_obs_target
        else:
            print("Info: cell_types_list is None or empty. Skipping cell type column processing.")
        data.data.obs['cell_type_num'] = 0
        if cell_types_list is not None:
            try:

                existing_ct_cols = [col for col in cell_types_list if col in target_adata.obs.columns]
                if not existing_ct_cols:
                    raise ValueError("None of the specified cell_types_list columns exist in target_adata.obs.")

                target_adata.obs['cell_type_num'] = (target_adata.obs[existing_ct_cols] > 0).sum(axis=1)
                print("Info: 'cell_type_num' calculated.")
            except Exception as e_calc:
                print(f"Warning: Could not calculate 'cell_type_num' ({e_calc}). Initializing with zeros.")
                target_adata.obs['cell_type_num'] = [0] * num_obs_target
        else:
            print("Info: cell_types_list is None or empty. Initializing 'cell_type_num' with zeros.")
            target_adata.obs['cell_type_num'] = [0] * num_obs_target


# update_anndata_obs(ST_adata_filter_norm, ST_adata_filter, cell_types)
@register_preprocessor("misc")
class CellTypeNum(BaseTransform):

    def __init__(self, split="pseudo", **kwargs):
        self.split = split
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        pseudo_adata_norm = data.get_split_data(split_name=self.split)
        cell_types = data.data.uns['cell_types_list']
        pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types] > 0).sum(axis=1)


@register_preprocessor("graph.cell")
class stdgcnGraph(BaseTransform):

    def __init__(self, inter_find_neighbor_method, inter_dist_method, inter_corr_dist_neighbors, spatial_link_method,
                 space_dist_threshold, real_intra_find_neighbor_method, real_intra_dist_method,
                 real_intra_pca_dimensionality_reduction, real_intra_corr_dist_neighbors, real_intra_dim,
                 pseudo_intra_find_neighbor_method, pseudo_intra_dist_method, pseudo_intra_corr_dist_neighbors,
                 pseudo_intra_pca_dimensionality_reduction, pseudo_intra_dim, real_split_name="test",
                 pseudo_split_name="pseudo", channel: Optional[str] = "feature.cell", channel_type: str = "obsm",
                 **kwargs):
        self.real_split_name = real_split_name
        self.pseudo_split_name = pseudo_split_name
        # 从 cli_args 构建 inter_exp_adj_paras
        self.inter_exp_adj_paras = {
            'find_neighbor_method': inter_find_neighbor_method,
            'dist_method': inter_dist_method,
            'corr_dist_neighbors': inter_corr_dist_neighbors,
        }
        # 从 cli_args 构建 spatial_adj_paras
        self.spatial_adj_paras = {
            'link_method': spatial_link_method,
            'space_dist_threshold': space_dist_threshold,
        }
        # 从 cli_args 构建 real_intra_exp_adj_paras
        self.real_intra_exp_adj_paras = {
            'find_neighbor_method': real_intra_find_neighbor_method,
            'dist_method': real_intra_dist_method,
            'corr_dist_neighbors': real_intra_corr_dist_neighbors,
            'PCA_dimensionality_reduction': real_intra_pca_dimensionality_reduction,
            'dim': real_intra_dim,
        }
        # 从 cli_args 构建 pseudo_intra_exp_adj_paras
        self.pseudo_intra_exp_adj_paras = {
            'find_neighbor_method': pseudo_intra_find_neighbor_method,
            'dist_method': pseudo_intra_dist_method,
            'corr_dist_neighbors': pseudo_intra_corr_dist_neighbors,
            'PCA_dimensionality_reduction': pseudo_intra_pca_dimensionality_reduction,
            'dim': pseudo_intra_dim,
        }
        self.channel = channel
        self.channel_type = channel_type

        super().__init__(**kwargs)

    def __call__(self, data: Data):
        ST_integration = data.data.obsm['DataInteragraionTransform']
        ST_adata_filter_norm = data.get_split_data(split_name=self.real_split_name)
        pseudo_adata_norm = data.get_split_data(split_name=self.pseudo_split_name)
        ST_adata_filter = data.data.raw.to_adata()[data.get_split_idx(self.real_split_name)]
        pseudo_adata_filter = data.data.raw.to_adata()[data.get_split_idx(self.pseudo_split_name)]

        A_inter_exp = inter_adj(
            ST_integration,
            find_neighbor_method=self.inter_exp_adj_paras['find_neighbor_method'],
            dist_method=self.inter_exp_adj_paras['dist_method'],
            corr_dist_neighbors=self.inter_exp_adj_paras['corr_dist_neighbors'],
        )

        A_intra_space = intra_dist_adj(
            ST_adata_filter_norm,
            link_method=self.spatial_adj_paras['link_method'],
            space_dist_threshold=self.spatial_adj_paras['space_dist_threshold'],
        )

        A_real_intra_exp = intra_exp_adj(
            ST_adata_filter_norm, find_neighbor_method=self.real_intra_exp_adj_paras['find_neighbor_method'],
            dist_method=self.real_intra_exp_adj_paras['dist_method'],
            PCA_dimensionality_reduction=self.real_intra_exp_adj_paras['PCA_dimensionality_reduction'],
            corr_dist_neighbors=self.real_intra_exp_adj_paras['corr_dist_neighbors'], channel=self.channel,
            channel_type=self.channel_type)

        A_pseudo_intra_exp = intra_exp_adj(
            pseudo_adata_norm, find_neighbor_method=self.pseudo_intra_exp_adj_paras['find_neighbor_method'],
            dist_method=self.pseudo_intra_exp_adj_paras['dist_method'],
            PCA_dimensionality_reduction=self.pseudo_intra_exp_adj_paras['PCA_dimensionality_reduction'],
            corr_dist_neighbors=self.pseudo_intra_exp_adj_paras['corr_dist_neighbors'], channel=self.channel,
            channel_type=self.channel_type)

        real_num = ST_adata_filter.shape[0]
        pseudo_num = pseudo_adata_filter.shape[0]

        adj_inter_exp = A_inter_exp.values
        adj_pseudo_intra_exp = A_intra_transfer(A_pseudo_intra_exp, 'pseudo', real_num, pseudo_num)
        adj_real_intra_exp = A_intra_transfer(A_real_intra_exp, 'real', real_num, pseudo_num)
        adj_intra_space = A_intra_transfer(A_intra_space, 'real', real_num, pseudo_num)

        adj_alpha = 1
        adj_beta = 1
        diag_power = 20
        adj_balance = (1 + adj_alpha + adj_beta) * diag_power
        adj_exp = torch.tensor(adj_inter_exp + adj_alpha * adj_pseudo_intra_exp +
                               adj_beta * adj_real_intra_exp) / adj_balance + torch.eye(adj_inter_exp.shape[0])
        adj_sp = torch.tensor(adj_intra_space) / diag_power + torch.eye(adj_intra_space.shape[0])

        norm = True
        if (norm == True):
            adj_exp = torch.tensor(adj_normalize(adj_exp, symmetry=True))
            adj_sp = torch.tensor(adj_normalize(adj_sp, symmetry=True))
        data.data.uns['adj_exp'] = adj_exp
        data.data.uns['adj_sp'] = adj_sp


@register_preprocessor("data", "interagration")
class DataInteragraionTransform(BaseTransform):

    def __init__(self, real_split_name="test", pseudo_split_name="pseudo", batch_removal_method="combat",
                 dimensionality_reduction_method='PCA', min_dim=50, scale=True, autoencoder_epoches=2000,
                 autoencoder_LR=1e-3, autoencoder_drop=0, cpu_num=-1, AE_device='GPU',
                 channel: Optional[str] = "feature.cell", channel_type: str = "obsm", **kwargs):
        self.real_split_name = real_split_name
        self.pseudo_split_name = pseudo_split_name
        self.batch_removal_method = batch_removal_method
        self.dimensionality_reduction_method = dimensionality_reduction_method
        self.min_dim = min_dim
        self.scale = scale
        self.autoencoder_epoches = autoencoder_epoches
        self.autoencoder_LR = autoencoder_LR
        self.autoencoder_drop = autoencoder_drop
        self.cpu_num = cpu_num
        self.AE_device = AE_device
        self.channel = channel
        self.channel_type = channel_type
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        real = data.get_split_data(split_name=self.real_split_name)
        pseudo = data.get_split_data(split_name=self.pseudo_split_name)
        dim = min(self.min_dim, int(real.shape[1] / 2))
        #remove split ref
        if self.batch_removal_method == 'mnn':
            mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True,
                                             var_subset=None)
            adata = mnn[0]
            if self.dimensionality_reduction_method == 'PCA':
                if self.scale == True:
                    sc.pp.scale(adata, max_value=None, zero_center=True)
                sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
                table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == 'autoencoder':
                data = torch.tensor(adata.X)
                x_size = data.shape[1]
                latent_size = dim
                hidden_size = int((x_size + latent_size) / 2)
                nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                                   p_drop=self.autoencoder_drop)
                optimizer_ae = torch.optim.Adam(nets.parameters(), lr=self.autoencoder_LR)
                loss_ae = nn.MSELoss(reduction='mean')
                embedding = auto_train(model=nets, epoch_n=self.autoencoder_epoches, loss_fn=loss_ae,
                                       optimizer=optimizer_ae, data=data, cpu_num=self.cpu_num,
                                       device=self.AE_device).detach().numpy()
                if self.scale == True:
                    embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
                table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == 'nmf':
                nmf = NMF(n_components=dim).fit_transform(adata.X)
                if self.scale == True:
                    nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
                table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == None:
                if self.scale == True:
                    sc.pp.scale(adata, max_value=None, zero_center=True)
                table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index],
                                     columns=adata.var.index.values)
            table = table.iloc[pseudo.shape[0]:, :].append(table.iloc[:pseudo.shape[0], :])
            table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

        elif self.batch_removal_method == 'scanorama':
            import scanorama
            scanorama.integrate_scanpy([real, pseudo], dimred=dim)
            table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
            table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
            table = table1.append(table2)
            table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

        elif self.batch_removal_method == 'combat':
            aaa = real.copy()
            aaa.obs = pd.DataFrame(index=aaa.obs.index)
            bbb = pseudo.copy()
            bbb.obs = pd.DataFrame(index=bbb.obs.index)
            adata = aaa.concatenate(bbb, batch_key='real_pseudo')
            sc.pp.combat(adata, key='real_pseudo')
            if self.dimensionality_reduction_method == 'PCA':
                if self.scale == True:
                    sc.pp.scale(adata, max_value=None, zero_center=True)
                sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
                table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == 'autoencoder':
                data = torch.tensor(adata.X)
                x_size = data.shape[1]
                latent_size = dim
                hidden_size = int((x_size + latent_size) / 2)
                nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                                   p_drop=self.autoencoder_drop)
                optimizer_ae = torch.optim.Adam(nets.parameters(), lr=self.autoencoder_LR)
                loss_ae = nn.MSELoss(reduction='mean')
                embedding = auto_train(model=nets, epoch_n=self.autoencoder_epoches, loss_fn=loss_ae,
                                       optimizer=optimizer_ae, data=data, cpu_num=self.cpu_num,
                                       device=self.AE_device).detach().numpy()
                if self.scale == True:
                    embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
                table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == 'nmf':
                nmf = NMF(n_components=dim).fit_transform(adata.X)
                if self.scale == True:
                    nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
                table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == None:
                if self.scale == True:
                    sc.pp.scale(adata, max_value=None, zero_center=True)
                table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index],
                                     columns=adata.var.index.values)
            table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

        else:
            aaa = real.copy()
            aaa.obs = pd.DataFrame(index=aaa.obs.index)
            bbb = pseudo.copy()
            bbb.obs = pd.DataFrame(index=bbb.obs.index)
            adata = aaa.concatenate(bbb, batch_key='real_pseudo')
            if self.dimensionality_reduction_method == 'PCA':
                if self.scale == True:
                    sc.pp.scale(adata, max_value=None, zero_center=True)
                sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
                table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == 'autoencoder':
                data = torch.tensor(adata.X)
                x_size = data.shape[1]
                latent_size = dim
                hidden_size = int((x_size + latent_size) / 2)
                nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                                   p_drop=self.autoencoder_drop)
                optimizer_ae = torch.optim.Adam(nets.parameters(), lr=self.autoencoder_LR)
                loss_ae = nn.MSELoss(reduction='mean')
                embedding = auto_train(model=nets, epoch_n=self.autoencoder_epoches, loss_fn=loss_ae,
                                       optimizer=optimizer_ae, data=data, cpu_num=self.cpu_num,
                                       device=self.AE_device).detach().numpy()
                if self.scale == True:
                    embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
                table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == 'nmf':
                nmf = NMF(n_components=dim).fit_transform(adata.X)
                if self.scale == True:
                    nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
                table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
            elif self.dimensionality_reduction_method == None:
                if self.scale == True:
                    sc.pp.scale(adata, max_value=None, zero_center=False)
                x = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
                # table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
                # table = pd.DataFrame(x, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
                table = pd.DataFrame(x, index=[str(i)[:-2] for i in adata.obs.index])
            table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

        table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist() + pseudo.obs['cell_num'].values.tolist())
        table.insert(2, 'cell_type_num',
                     real.obs['cell_type_num'].values.tolist() + pseudo.obs['cell_type_num'].values.tolist())

        data.data.obsm[self.out] = table


from typing import Any, List, Literal, Mapping, Optional, Tuple, Union


class stdGCNWrapper(BaseRegressionMethod):

    def __init__(self, cli_args):
        # 从 cli_args 构建 integration_for_adj_paras

        # 从 cli_args 构建 GCN_paras
        self.GCN_paras = {
            'epoch_n': cli_args.epoch_n,
            'dim': cli_args.gcn_dim,  # 假设 GCN_paras 内部期望的键是 'dim'
            'common_hid_layers_num': cli_args.common_hid_layers_num,
            'fcnn_hid_layers_num': cli_args.fcnn_hid_layers_num,
            'dropout': cli_args.dropout,
            'learning_rate_SGD': cli_args.learning_rate_sgd,  # 假设 GCN_paras 内部期望的键是 'learning_rate_SGD'
            'weight_decay_SGD': cli_args.weight_decay_sgd,  # 假设 GCN_paras 内部期望的键是 'weight_decay_SGD'
            'momentum': cli_args.momentum,
            'dampening': cli_args.dampening,
            'nesterov': cli_args.nesterov,
            'early_stopping_patience': cli_args.early_stopping_patience,
            'clip_grad_max_norm': cli_args.clip_grad_max_norm,
            'print_loss_epoch_step': cli_args.print_loss_epoch_step,
        }
        self.GCN_device = cli_args.gcn_device  # 直接从 cli_args 获取
        self.n_jobs = cli_args.n_jobs  # 直接从 cli_args 获取

    # @staticmethod
    # def preprocessing_pipeline(args,log_level: LogLevel = "INFO"):
    #     return Compose(
    #     CelltypeTransform(),
    #     pseudoSpotGen(
    #         spot_num=args.spot_num,  # 从 pseudo_spot_simulation_paras 映射过来
    #         min_cell_number_in_spot=args.min_cell_num_in_spot, # 从 pseudo_spot_simulation_paras 映射过来 (注意原始key是 min_cell_num_in_spot)
    #         max_cell_number_in_spot=args.max_cell_num_in_spot, # 从 pseudo_spot_simulation_paras 映射过来 (注意原始key是 max_cell_num_in_spot)
    #         max_cell_types_in_spot=args.max_cell_types_in_spot, # 从 pseudo_spot_simulation_paras 映射过来
    #         generation_method=args.generation_method,  # 从 pseudo_spot_simulation_paras 映射过来
    #         n_jobs=args.n_jobs  # 直接从 args 获取
    #     ),
    #     RemoveSplit(split_name="ref", log_level="INFO"), # 无变化
    #     FilterGenesCommon(split_keys=["pseudo", "test"]), # 无变化
    #     FilterGenesPlaceHolder(),
    #     SaveRaw(), # 无变化
    #     NormalizeTotalLog1P(),
    #     FilterGenesTopK(num_genes=4000),
    #     updateAnndataObsTransform(split="test"), # 无变化
    #     CellTypeNum(), # 无变化
    #     CellPCA(out="feature.cell"),
    #     DataInteragraionTransform(batch_removal_method=args.adj_batch_removal_method if args.adj_batch_removal_method and args.adj_batch_removal_method.lower() != 'none' else None,
    #         min_dim=args.adj_dim,
    #         dimensionality_reduction_method=args.adj_dimensionality_reduction_method if args.adj_dimensionality_reduction_method and args.adj_dimensionality_reduction_method.lower() != 'none' else None,
    #         scale=args.adj_scale, cpu_num=args.n_jobs , AE_device=args.gcn_device),
    #     stdgcnGraph(
    #         inter_find_neighbor_method=args.inter_find_neighbor_method,
    #         inter_dist_method=args.inter_dist_method,
    #         inter_corr_dist_neighbors=args.inter_corr_dist_neighbors,
    #         spatial_link_method=args.spatial_link_method,
    #         space_dist_threshold=args.space_dist_threshold,
    #         real_intra_find_neighbor_method=args.real_intra_find_neighbor_method,
    #         real_intra_dist_method=args.real_intra_dist_method,
    #         real_intra_pca_dimensionality_reduction=args.real_intra_pca_dimensionality_reduction,
    #         real_intra_corr_dist_neighbors=args.real_intra_corr_dist_neighbors,
    #         real_intra_dim=args.real_intra_dim,
    #         pseudo_intra_find_neighbor_method=args.pseudo_intra_find_neighbor_method,
    #         pseudo_intra_dist_method=args.pseudo_intra_dist_method,
    #         pseudo_intra_corr_dist_neighbors=args.pseudo_intra_corr_dist_neighbors,
    #         pseudo_intra_pca_dimensionality_reduction=args.pseudo_intra_pca_dimensionality_reduction,
    #         pseudo_intra_dim=args.pseudo_intra_dim,
    #         ),
    #     DataInteragraionTransform(batch_removal_method=args.feat_batch_removal_method if args.feat_batch_removal_method and args.feat_batch_removal_method.lower() != 'none' else None,
    #         min_dim=args.feat_dim,
    #         dimensionality_reduction_method=args.feat_dimensionality_reduction_method if args.feat_dimensionality_reduction_method and args.feat_dimensionality_reduction_method.lower() != 'none' else None,
    #         scale=args.feat_scale, cpu_num=args.n_jobs , AE_device=args.gcn_device),
    #     SetConfig({"label_channel": "cell_type_portion"}))
    @staticmethod
    def preprocessing_pipeline(args, log_level: LogLevel = "INFO"):
        return Compose(
            CelltypeTransform(),
            stdGCNMarkGenes(
                preprocess=args.fmg_preprocess,
                highly_variable_genes=args.fmg_highly_variable_genes,
                PCA_components=args.fmg_pca_components,
                filter_wilcoxon_marker_genes=args.fmg_filter_wilcoxon_marker_genes,
                marker_gene_method=args.marker_gene_method,
                pvals_adj_threshold=args.pvals_adj_threshold,
                log_fold_change_threshold=args.log_fold_change_threshold,
                min_within_group_fraction_threshold=args.
                min_within_group_fraction_threshold,  # 从 find_marker_genes_paras 映射过来
                max_between_group_fraction_threshold=args.
                max_between_group_fraction_threshold,  # 从 find_marker_genes_paras 映射过来
                top_gene_per_type=args.top_gene_per_type  # 从 find_marker_genes_paras 映射过来
            ),
            pseudoSpotGen(
                spot_num=args.spot_num,  # 从 pseudo_spot_simulation_paras 映射过来
                min_cell_number_in_spot=args.
                min_cell_num_in_spot,  # 从 pseudo_spot_simulation_paras 映射过来 (注意原始key是 min_cell_num_in_spot)
                max_cell_number_in_spot=args.
                max_cell_num_in_spot,  # 从 pseudo_spot_simulation_paras 映射过来 (注意原始key是 max_cell_num_in_spot)
                max_cell_types_in_spot=args.max_cell_types_in_spot,  # 从 pseudo_spot_simulation_paras 映射过来
                generation_method=args.generation_method,  # 从 pseudo_spot_simulation_paras 映射过来
                n_jobs=args.n_jobs  # 直接从 args 获取
            ),
            RemoveSplit(split_name="ref", log_level="INFO"),  # 无变化
            FilterGenesCommon(split_keys=["pseudo", "test"]),  # 无变化
            SaveRaw(),  # 无变化
            STPreprocessTransform(
                normalize=args.dn_normalize,  # 从 data_normalization_paras 映射过来
                log=args.dn_log,  # 从 data_normalization_paras 映射过来
                scale=args.dn_scale,  # 从 data_normalization_paras 映射过来
                split="pseudo"),
            updateAnndataObsTransform(split="test"),  # 无变化
            STPreprocessTransform(
                normalize=args.dn_normalize,  # 从 data_normalization_paras 映射过来
                log=args.dn_log,  # 从 data_normalization_paras 映射过来
                scale=args.dn_scale,  # 从 data_normalization_paras 映射过来
                split="test"),
            CellTypeNum(),  # 无变化
            FeatureCellPlaceHolder(out="feature.cell"),
            DataInteragraionTransform(
                batch_removal_method=args.adj_batch_removal_method
                if args.adj_batch_removal_method and args.adj_batch_removal_method.lower() != 'none' else None,
                min_dim=args.adj_dim, dimensionality_reduction_method=args.adj_dimensionality_reduction_method if
                args.adj_dimensionality_reduction_method and args.adj_dimensionality_reduction_method.lower() != 'none'
                else None, scale=args.adj_scale, cpu_num=args.n_jobs, AE_device=args.gcn_device),
            stdgcnGraph(
                inter_find_neighbor_method=args.inter_find_neighbor_method,
                inter_dist_method=args.inter_dist_method,
                inter_corr_dist_neighbors=args.inter_corr_dist_neighbors,
                spatial_link_method=args.spatial_link_method,
                space_dist_threshold=args.space_dist_threshold,
                real_intra_find_neighbor_method=args.real_intra_find_neighbor_method,
                real_intra_dist_method=args.real_intra_dist_method,
                real_intra_pca_dimensionality_reduction=args.real_intra_pca_dimensionality_reduction,
                real_intra_corr_dist_neighbors=args.real_intra_corr_dist_neighbors,
                real_intra_dim=args.real_intra_dim,
                pseudo_intra_find_neighbor_method=args.pseudo_intra_find_neighbor_method,
                pseudo_intra_dist_method=args.pseudo_intra_dist_method,
                pseudo_intra_corr_dist_neighbors=args.pseudo_intra_corr_dist_neighbors,
                pseudo_intra_pca_dimensionality_reduction=args.pseudo_intra_pca_dimensionality_reduction,
                pseudo_intra_dim=args.pseudo_intra_dim,
            ),
            DataInteragraionTransform(
                batch_removal_method=args.feat_batch_removal_method
                if args.feat_batch_removal_method and args.feat_batch_removal_method.lower() != 'none' else None,
                min_dim=args.feat_dim, dimensionality_reduction_method=args.feat_dimensionality_reduction_method
                if args.feat_dimensionality_reduction_method
                and args.feat_dimensionality_reduction_method.lower() != 'none' else None, scale=args.feat_scale,
                cpu_num=args.n_jobs, AE_device=args.gcn_device),
            SetConfig({"label_channel": "cell_type_portion"}))

    def fit_score(self, x, y, *, score_func: Optional[Union[str, Mapping[Any,
                                                                         float]]] = None, return_pred: bool = False,
                  valid_idx=None, test_idx=None, **fit_kwargs) -> Union[float, Tuple[float, Any]]:
        """Shortcut for fitting data using the input feature and return eval.

        Note
        ----
        Only work for models where the fitting does not require labeled data, i.e. unsupervised methods.

        """
        self.fit(x, y, **fit_kwargs)
        return self.score(x, y, score_func=score_func, return_pred=return_pred, valid_idx=valid_idx, test_idx=test_idx)

    def score(self, x, y, *, score_func: Optional[Union[str, Mapping[Any, float]]] = None, return_pred: bool = False,
              valid_idx=None, test_idx=None) -> Union[float, Tuple[float, Any]]:
        y_pred = self.predict(x)
        func = resolve_score_func(score_func or self._DEFAULT_METRIC)
        if valid_idx is None:
            score = func(y, y_pred)
            return (score, y_pred) if return_pred else score
        else:
            valid_score = func(y[valid_idx], y_pred[valid_idx])
            test_score = func(y[test_idx], y_pred[test_idx])
            return (valid_score, test_score, y_pred) if return_pred else (valid_score, test_score)

    def fit(
        self,
        inputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        y: Optional[Any] = None,
    ):
        ST_adata_filter_norm, ST_integration_batch_removed, adj_exp, adj_sp, word_to_idx_celltype, train_idx, valid_idx, test_idx = inputs
        # ST_adata_filter_norm.obs['coor_X'] = spatial['x']
        # ST_adata_filter_norm.obs['coor_Y'] = spatial['y']
        # ST_integration = data_integration(
        #     ST_adata_filter_norm, pseudo_adata_norm,
        #    )

        # ST_integration_batch_removed = data_integration(
        #     ST_adata_filter_norm, pseudo_adata_norm,
        #     batch_removal_method=self.integration_for_feature_paras['batch_removal_method'],
        #     dim=min(int(ST_adata_filter_norm.shape[1] * 1 / 2), self.integration_for_feature_paras['dim']),
        #     dimensionality_reduction_method=self.integration_for_feature_paras['dimensionality_reduction_method'],
        #     scale=self.integration_for_feature_paras['scale'], cpu_num=self.n_jobs, AE_device=self.GCN_device)
        feature = torch.tensor(ST_integration_batch_removed.iloc[:, 3:].values)

        input_layer = feature.shape[1]
        hidden_layer = min(int(ST_adata_filter_norm.shape[1] * 1 / 2), self.GCN_paras['dim'])
        output_layer1 = len(word_to_idx_celltype)
        epoch_n = self.GCN_paras['epoch_n']
        common_hid_layers_num = self.GCN_paras['common_hid_layers_num']
        fcnn_hid_layers_num = self.GCN_paras['fcnn_hid_layers_num']
        dropout = self.GCN_paras['dropout']
        learning_rate_SGD = self.GCN_paras['learning_rate_SGD']
        weight_decay_SGD = self.GCN_paras['weight_decay_SGD']
        momentum = self.GCN_paras['momentum']
        dampening = self.GCN_paras['dampening']
        nesterov = self.GCN_paras['nesterov']
        early_stopping_patience = self.GCN_paras['early_stopping_patience']
        clip_grad_max_norm = self.GCN_paras['clip_grad_max_norm']
        LambdaLR_scheduler_coefficient = 0.997
        ReduceLROnPlateau_factor = 0.1
        ReduceLROnPlateau_patience = 5
        scheduler = 'scheduler_ReduceLROnPlateau'
        print_epoch_step = self.GCN_paras['print_loss_epoch_step']
        cpu_num = self.n_jobs

        model = conGCN(nfeat=input_layer, nhid=hidden_layer, common_hid_layers_num=common_hid_layers_num,
                       fcnn_hid_layers_num=fcnn_hid_layers_num, dropout=dropout, nout1=output_layer1)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_SGD, momentum=momentum,
                                    weight_decay=weight_decay_SGD, dampening=dampening, nesterov=nesterov)

        scheduler_LambdaLR = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: LambdaLR_scheduler_coefficient**epoch)
        scheduler_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                                 factor=ReduceLROnPlateau_factor,
                                                                                 patience=ReduceLROnPlateau_patience,
                                                                                 threshold=0.0001, threshold_mode='rel',
                                                                                 cooldown=0, min_lr=0)
        if scheduler == 'scheduler_LambdaLR':
            scheduler = scheduler_LambdaLR
        elif scheduler == 'scheduler_ReduceLROnPlateau':
            scheduler = scheduler_ReduceLROnPlateau
        else:
            scheduler = None

        loss_fn1 = nn.KLDivLoss(reduction='mean')

        adjs = [adj_exp.float(), adj_sp.float()]

        output1, loss, trained_model = conGCN_train(
            model=model, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, feature=feature, adjs=adjs,
            label=y, epoch_n=epoch_n, loss_fn=loss_fn1, optimizer=optimizer, scheduler=scheduler,
            early_stopping_patience=early_stopping_patience, clip_grad_max_norm=clip_grad_max_norm,
            print_epoch_step=print_epoch_step, cpu_num=cpu_num, GCN_device=self.GCN_device)

        loss_table = pd.DataFrame(loss, columns=['train', 'valid', 'test'])

        # predict_table = pd.DataFrame(
        #     np.exp(output1[:test_len].detach().numpy()).tolist(), index=ST_adata_filter_norm.obs.index,
        #     columns=pseudo_adata_norm.obs.columns[:-2])
        # predict_table.to_csv(output_path+'/predict_result.csv', index=True, header=True)

        # torch.save(trained_model, output_path+'/model_parameters')

        # pred_use = np.round_(output1.exp().detach()[:test_len], decimals=4)
        # # cell_type_list = cell_types
        # coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]

        # ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].detach().numpy())

        # torch.cuda.empty_cache()

        # return ST_adata_filter_norm
        # self.predict_result=np.exp(output1[:test_len].detach().numpy())
        self.all_predict_result = np.exp(output1.detach().numpy())

    def predict(self, x: Optional[Any] = None):
        return self.all_predict_result


# inputs, y = data.get_data(split_name="test", return_type="torch")
# x, spatial = inputs
# ref_count = data.get_feature(split_name="ref", return_type="numpy")
# ref_annot = data.get_feature(split_name="ref", return_type="numpy", channel="cellType", channel_type="obs")
# sc_adata = sc.read_csv(sc_path+"/sc_data.tsv", delimiter='\t')
# sc_label = pd.read_table(sc_path+"/sc_label.tsv", sep = '\t', header = 0, index_col = 0, encoding = "utf-8")
# sc_label.columns = ['cell_type']
# sc_adata.obs['cell_type'] = sc_label['cell_type'].values

# sc_adata_data=Data(data=sc_adata)
# stdgcn_mark_genes =
# stdGCNMarkGenes(sc_adata_data)
# selected_genes, cell_type_marker_genes=sc_adata_data.uns['gene_list'],sc_adata_data.uns['gene_dict']

# with open(output_path+"/marker_genes.tsv", 'w') as f:
#     for gene in selected_genes:
#         f.write(str(gene) + '\n')

# print("{} genes have been selected as marker genes.".format(len(selected_genes)))
# n_jobs = 10
# sc_adata_data=Data(data=sc_adata)
# pseudospotgen =
# pseudospotgen(sc_adata_data)
# pseudo_adata=sc_adata_data.uns['pseudoSpotGen']

# ST_adata = sc.read_csv(ST_path+"/ST_data.tsv", delimiter='\t')
# ST_coor = pd.read_table(ST_path+"/coordinates.csv", sep = ',', header = 0, index_col = 0, encoding = "utf-8")

# ST_adata = ad.AnnData(X=x)
# ST_adata.obs['coor_X'] = spatial['x']
# ST_adata.obs['coor_Y'] = spatial['y']
# ST_groundtruth = y
# assert y.columns == cell_types
# for i in cell_types:
#     ST_adata.obs[i] = ST_groundtruth[i]

# ST_genes = ST_adata.var.index.values
# pseudo_genes = pseudo_adata.var.index.values
# common_genes = set(ST_genes).intersection(set(pseudo_genes))
# ST_adata_filter = ST_adata[:, list(common_genes)]
# pseudo_adata_filter = pseudo_adata[:, list(common_genes)]

# st_preprocess_1 =
# pseudo_adata_filter_data=Data(data=pseudo_adata_filter)
# st_preprocess_1(pseudo_adata_filter_data)
# pseudo_adata_norm=pseudo_adata_filter_data.data

# st_preprocess_2 =
# ST_adata_filter_data=Data(data=ST_adata_filter)
# st_preprocess_2(ST_adata_filter_data)
# ST_adata_filter_norm=ST_adata_filter_data.data[:, ]
