# Python Standard Library
import copy
import math
import multiprocessing
import pickle
import random
import time

# Third-party libraries
import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# PyTorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric, KDTree, NearestNeighbors  # 合并了来自 sklearn.neighbors 的导入
from torch.autograd import Variable
from torch.nn.modules.module import Module  # 注意: 通常直接使用 nn.Module
from torch.nn.parameter import Parameter  # 注意: 通常直接使用 nn.Parameter

# tqdm
from tqdm.notebook import tqdm

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.base import BaseRegressionMethod
from dance.modules.spatial.cell_type_deconvo.dstg import GCN
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel


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


def conGCN_train(model, train_valid_len, test_len, feature, adjs, label, epoch_n, loss_fn, optimizer,
                 train_valid_ratio=0.9, scheduler=None, early_stopping_patience=5, clip_grad_max_norm=1,
                 load_test_groundtruth=False, print_epoch_step=1, cpu_num=-1, GCN_device='CPU'):

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

    train_idx = range(int(train_valid_len * train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)

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

        loss_train1 = loss_fn(output1[list(np.array(train_idx) + test_len)],
                              label[list(np.array(train_idx) + test_len)].float())
        loss_val1 = loss_fn(output1[list(np.array(valid_idx) + test_len)],
                            label[list(np.array(valid_idx) + test_len)].float())
        if load_test_groundtruth == True:
            loss_test1 = loss_fn(output1[:test_len], label[:test_len].float())
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

    knn.fit(ST_exp.obs[['coor_X', 'coor_Y']])
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
):

    ST_exp = adata.copy()

    sc.pp.scale(ST_exp, max_value=None, zero_center=True)
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
        sc.pp.scale(ST_exp, max_value=None, zero_center=True)
        input_data = ST_exp.X
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


def ST_preprocess(ST_exp, normalize=True, log=True, highly_variable_genes=False, regress_out=False, scale=False,
                  scale_max_value=None, scale_zero_center=True, hvg_min_mean=0.0125, hvg_max_mean=3, hvg_min_disp=0.5,
                  highly_variable_gene_num=None):

    adata = ST_exp.copy()

    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if log == True:
        sc.pp.log1p(adata)

    adata.layers['scale.data'] = adata.X.copy()

    if highly_variable_genes == True:
        sc.pp.highly_variable_genes(
            adata,
            min_mean=hvg_min_mean,
            max_mean=hvg_max_mean,
            min_disp=hvg_min_disp,
            n_top_genes=highly_variable_gene_num,
        )
        adata = adata[:, adata.var.highly_variable]

    if regress_out == True:
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        sc.pp.filter_cells(adata, min_counts=0)
        sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

    if scale == True:
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=scale_zero_center)

    return adata


def find_marker_genes(
    sc_exp,
    preprocess=True,
    highly_variable_genes=True,
    regress_out=False,
    scale=False,
    PCA_components=50,
    marker_gene_method='wilcoxon',
    filter_wilcoxon_marker_genes=True,
    top_gene_per_type=20,
    pvals_adj_threshold=0.10,
    log_fold_change_threshold=1,
    min_within_group_fraction_threshold=0.7,
    max_between_group_fraction_threshold=0.3,
):

    if preprocess == True:
        sc_adata_marker_gene = ST_preprocess(
            sc_exp.copy(),
            normalize=True,
            log=True,
            highly_variable_genes=highly_variable_genes,
            regress_out=regress_out,
            scale=scale,
        )
    else:
        sc_adata_marker_gene = sc_exp.copy()

    sc.tl.pca(sc_adata_marker_gene, n_comps=PCA_components, svd_solver='arpack', random_state=None)

    layer = 'scale.data'
    sc.tl.rank_genes_groups(sc_adata_marker_gene, 'cell_type', layer=layer, use_raw=False, pts=True,
                            method=marker_gene_method, corr_method='benjamini-hochberg', key_added=marker_gene_method)

    if marker_gene_method == 'wilcoxon':
        if filter_wilcoxon_marker_genes == True:
            gene_dict = {}
            gene_list = []
            for name in sc_adata_marker_gene.obs['cell_type'].unique():
                data = sc.get.rank_genes_groups_df(sc_adata_marker_gene, group=name,
                                                   key=marker_gene_method).sort_values('pvals_adj')
                if pvals_adj_threshold != None:
                    data = data[data['pvals_adj'] < pvals_adj_threshold]
                if log_fold_change_threshold != None:
                    data = data[data['logfoldchanges'] >= log_fold_change_threshold]
                if min_within_group_fraction_threshold != None:
                    data = data[data['pct_nz_group'] >= min_within_group_fraction_threshold]
                if max_between_group_fraction_threshold != None:
                    data = data[data['pct_nz_reference'] < max_between_group_fraction_threshold]
                gene_dict[name] = data['names'].values[:top_gene_per_type].tolist()
                gene_list = gene_list + data['names'].values[:top_gene_per_type].tolist()
                gene_list = list(set(gene_list))
        else:
            gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
            gene_dict = {}
            for i in gene_table.columns:
                gene_dict[i] = gene_table[i].values.tolist()
            gene_list = list({item for sublist in gene_table.values.tolist() for item in sublist})
    elif marker_gene_method == 'logreg':
        gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
        gene_dict = {}
        for i in gene_table.columns:
            gene_dict[i] = gene_table[i].values.tolist()
        gene_list = list({item for sublist in gene_table.values.tolist() for item in sublist})
    else:
        print("marker_gene_method should be 'logreg' or 'wilcoxon'")

    return gene_list, gene_dict


def generate_a_spot(
    sc_exp,
    min_cell_number_in_spot,
    max_cell_number_in_spot,
    max_cell_types_in_spot,
    generation_method,
):

    if generation_method == 'cell':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]
    elif generation_method == 'celltype':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_type_list = list(sc_exp.obs['cell_type'].unique())
        cell_type_num = random.randint(1, max_cell_types_in_spot)

        while (True):
            cell_type_list_selected = random.choices(sc_exp.obs['cell_type'].value_counts().keys(), k=cell_type_num)
            if len(set(cell_type_list_selected)) == cell_type_num:
                break
        sc_exp_filter = sc_exp[sc_exp.obs['cell_type'].isin(cell_type_list_selected)]

        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for i in picked_cell_type:
            data = sc_exp[sc_exp.obs['cell_type'] == i]
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])

        return sc_exp_filter[picked_cells]
    else:
        print('generation_method should be "cell" or "celltype" ')


def pseudo_spot_generation(sc_exp, idx_to_word_celltype, spot_num, min_cell_number_in_spot, max_cell_number_in_spot,
                           max_cell_types_in_spot, generation_method, n_jobs=-1):

    cell_type_num = len(sc_exp.obs['cell_type'].unique())

    cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        pool = multiprocessing.Pool(processes=cores)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
    args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot, max_cell_types_in_spot, generation_method)
            for i in range(spot_num)]
    generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))

    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            pseudo_fraction_table[i, type_idx] += 1
    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = ad.AnnData(X=pseudo_spots_table.iloc[:, :].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i] / pseudo_fraction_table['cell_num']
    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)

    return pseudo_spots


def data_integration(real, pseudo, batch_removal_method="combat", dimensionality_reduction_method='PCA', dim=50,
                     scale=True, autoencoder_epoches=2000, autoencoder_LR=1e-3, autoencoder_drop=0, cpu_num=-1,
                     AE_device='GPU'):

    if batch_removal_method == 'mnn':
        mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True,
                                         var_subset=None)
        adata = mnn[0]
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size) / 2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                               p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction='mean')
            embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
                                   data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
            if scale == True:
                embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table = table.iloc[pseudo.shape[0]:, :].append(table.iloc[:pseudo.shape[0], :])
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    elif batch_removal_method == 'scanorama':
        import scanorama
        scanorama.integrate_scanpy([real, pseudo], dimred=dim)
        table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
        table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
        table = table1.append(table2)
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    elif batch_removal_method == 'combat':
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index=aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index=bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        sc.pp.combat(adata, key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size) / 2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                               p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction='mean')
            embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
                                   data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
            if scale == True:
                embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    else:
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index=aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index=bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size) / 2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                               p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction='mean')
            embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
                                   data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
            if scale == True:
                embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=False)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist() + pseudo.obs['cell_num'].values.tolist())
    table.insert(2, 'cell_type_num',
                 real.obs['cell_type_num'].values.tolist() + pseudo.obs['cell_type_num'].values.tolist())

    return table


# !/usr/bin/env python
# coding: utf-8

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
"""This module is used to provide the path of the loading data and saving data.

Parameters:
sc_path: The path for loading single cell reference data.
ST_path: The path for loading spatial transcriptomics data.
output_path: The path for saving output files.

The relevant file name and data format for loading:
sc_data.tsv: The expression matrix of the single cell reference data with cells as rows and genes as columns. This file should be saved in "sc_path".
sc_label.tsv: The cell-type annotation of sincle cell data. The table should have two columns: The cell barcode/name and the cell-type annotation information.
            This file should be saved in "sc_path".
ST_data.tsv: The expression matrix of the spatial transcriptomics data with spots as rows and genes as columns. This file should be saved in "ST_path".
coordinates.csv: The coordinates of the spatial transcriptomics data. The table should have three columns: Spot barcode/name, X axis (column name 'x'), and Y axis (column name 'y').
            This file should be saved in "ST_path".
marker_genes.tsv [optional]: The gene list used to run STdGCN. Each row is a gene and no table header is permitted. This file should be saved in "sc_path".
ST_ground_truth.tsv [optional]: The ground truth of ST data. The data should be transformed into the cell type proportions. This file should be saved in "ST_path".

"""
# paths = {
#     'sc_path': './data/sc_data',
#     'ST_path': './data/ST_data',
#     'output_path': './output',
# }
paths = {"dataset": "CARD_synthetic", "datadir": "data/spatial"}
"""This module is used to preprocess the input data and identify marker genes
[optional].

Parameters:
'preprocess': [bool]. Select whether the input expression data needs to be preprocessed. This step includes normalization, logarithmization, selecting highly variable genes,
                    regressing out mitochondrial genes, and scaling data.
'normalize': [bool]. When 'preprocess'=True, select whether you need to normalize each cell/spot by total counts = 10,000, so that every cell/spot has the same total
                    count after normalization.
'log': [bool]. When 'preprocess'=True, select whether you need to logarithmize (X=log(X+1)) the expression matrix.
'highly_variable_genes': [bool]. When 'preprocess'=True, select whether you need to filter the highly variable genes.
'highly_variable_gene_num': [int or None]. When 'preprocess'=True and 'highly_variable_genes'=True, select the number of highly-variable genes to keep.
'regress_out': [bool]. When 'preprocess'=True, select whether you need to regress out mitochondrial genes.
'scale': [bool]. When 'preprocess'=True, select whether you need to scale each gene to unit variance and zero mean.
'PCA_components': [int]. Number of principal components to compute for principal component analysis (PCA).
'marker_gene_method': ['logreg', 'wilcoxon']. We used "scanpy.tl.rank_genes_groups" (https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html)
                    to identify cell type marker genes. For marker gene selection, STdGCN provides two methods, 'wilcoxon' (Wilcoxon rank-sum) and 'logreg' (uses
                    logistic regression).
'top_gene_per_type': [int]. The number of genes for each cell type that can be used to train STdGCN.
'filter_wilcoxon_marker_genes': [bool]. When 'marker_gene_method'='wilcoxon', select whether you need additional steps for gene filtering.
'pvals_adj_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes with corrected p-values < 'pvals_adj_threshold' were kept.
'log_fold_change_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes with log fold change > 'log_fold_change_threshold' were kept.
'min_within_group_fraction_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes expressed with fraction at least
                    'min_within_group_fraction_threshold' in the cell type were kept.
'max_between_group_fraction_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes expressed with fraction at most
                    'max_between_group_fraction_threshold' in the union of the rest of cell types were kept.

"""
find_marker_genes_paras = {
    'preprocess': True,
    'normalize': True,
    'log': True,
    'highly_variable_genes': False,
    'highly_variable_gene_num': None,
    'regress_out': False,
    'PCA_components': 30,
    'marker_gene_method': 'logreg',
    'top_gene_per_type': 100,
    'filter_wilcoxon_marker_genes': True,
    'pvals_adj_threshold': 0.10,
    'log_fold_change_threshold': 1,
    'min_within_group_fraction_threshold': None,
    'max_between_group_fraction_threshold': None,
}
"""This module is used to simulate pseudo-spots.

Parameters:
'spot_num': [int]. The number of pseudo-spots.
'min_cell_num_in_spot': [int]. The minimum number of cells in a pseudo-spot.
'max_cell_num_in_spot': [int]. The maximum number of cells in a pseudo-spot.
'generation_method': ['cell' or 'celltype']. STdGCN provides two pseudo-spot simulation methods. When 'generation_method'='cell', each cell is equally selected. When
                    'generation_method'='celltype', each cell type is equally selected. See manuscript for more details.
'max_cell_types_in_spot': [int]. When 'generation_method'='celltype', choose the maximum number of cell types in a pseudo-spot.

"""
pseudo_spot_simulation_paras = {
    'spot_num': 30000,
    'min_cell_num_in_spot': 8,
    'max_cell_num_in_spot': 12,
    'generation_method': 'celltype',
    'max_cell_types_in_spot': 4,
}
"""This module is used for real- and pseudo- spots normalization.

Parameters:
'normalize': [bool]. Select whether you need to normalize each cell/spot by total counts = 10,000, so that every cell/spot has the same total count after normalization.
'log': [bool]. Select whether you need to logarithmize (X=log(X+1)) the expression matrix.
'scale': [bool]. Select whether you need to scale each gene to unit variance and zero mean.

"""
data_normalization_paras = {
    'normalize': True,
    'log': True,
    'scale': False,
}
"""This module is used to integrate the normalized real- and pseudo- spots together to
construct the real-to-pseudo-spot link graph.

Parameters:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. Considering batch effects, STdGCN provides four integration methods: mnn (mnnpy, DOI:10.1038/nbt.4091),
                    scanorama (Scanorama, DOI: 10.1038/s41587-019-0113-3), combat (Combat, DOI: 10.1093/biostatistics/kxj037), None (concatenation with no batch removal).
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. When 'batch_removal_method' is not 'scanorama', select whether the data needs dimensionality reduction, and which
                    dimensionality reduction method is applied.
'dim': [int]. When 'batch_removal_method'='scanorama', select the dimension for this method. When 'batch_removal_method' is not 'scanorama' and 'dimensionality_reduction_method' is
                    not None, select the dimension of the dimensionality reduction.
'scale': [bool]. When 'batch_removal_method' is not 'scanorama', select whether you need to scale each gene to unit variance and zero mean.

"""
integration_for_adj_paras = {
    'batch_removal_method': None,
    'dim': 30,
    'dimensionality_reduction_method': 'PCA',
    'scale': True,
}
"""The module is used to construct the adjacency matrix of the expression graph, which
contains three subgraphs: a real-to-pseudo-spot graph, a pseudo-spots internal graph,
and a real-spots internal graph.

Parameters:
'find_neighbor_method' ['MNN', 'KNN']. STdGCN provides two methods for link graph construction, KNN (K-nearest neighbors) and MNN (mutual nearest neighbors, DOI: 10.1038/nbt.4091).
'dist_method': ['euclidean', 'cosine']. The metrics used for computing paired distances between spots.
'corr_dist_neighbors': [int]. The number of nearest neighbors.
'PCA_dimensionality_reduction': [bool]. For pseudo-spots internal graph and real-spots internal graph construction, select if the data needs to use PCA dimensionality reduction before
                    computing paired distances between spots.
'dim': [int]. When 'PCA_dimensionality_reduction'=True, select the dimension of the PCA.

"""
inter_exp_adj_paras = {
    'find_neighbor_method': 'MNN',
    'dist_method': 'cosine',
    'corr_dist_neighbors': 20,
}
real_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN',
    'dist_method': 'cosine',
    'corr_dist_neighbors': 10,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}
pseudo_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN',
    'dist_method': 'cosine',
    'corr_dist_neighbors': 20,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}
"""The module is used to construct the adjacency matrix of the spatial graph.

Parameters:
'space_dist_threshold': [float or None]. Only the distance between two spots smaller than 'space_dist_threshold' can be linked.
'link_method' ['soft', 'hard']. If spot i and j linked, A(i,j)=1 if 'link_method'='hard', while A(i,j)=1/distance(i,j) if 'link_method'='soft'. See manuscript for more details.

"""
spatial_adj_paras = {
    'link_method': 'soft',
    'space_dist_threshold': 2,
}
"""This module is used to integrate the normalized real- and pseudo- spots as the input
feature for STdGCN.

Parameters:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. Considering batch effects, STdGCN provides four integration methods: mnn (mnnpy, DOI:10.1038/nbt.4091),
                    scanorama (Scanorama, DOI: 10.1038/s41587-019-0113-3), combat (Combat, DOI: 10.1093/biostatistics/kxj037), None (concatenation with no batch removal).
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. When 'batch_removal_method' is not 'scanorama', select whether the data needs dimensionality reduction, and which
                    dimensionality reduction method is applied.
'dim': [int]. When 'batch_removal_method'='scanorama', select the dimension for this method. When 'batch_removal_method' is not 'scanorama' and 'dimensionality_reduction_method' is
                    not None, select the dimension of the dimensionality reduction.
'scale': [bool]. When 'batch_removal_method' is not 'scanorama', select whether you need to scale each gene to unit variance and zero mean.

"""
integration_for_feature_paras = {
    'batch_removal_method': None,
    'dimensionality_reduction_method': None,
    'dim': 80,
    'scale': True,
}
"""This module is used for setting the deep learning parameters for STdGCN.

Parameters:
'epoch_n': [int]. The maximum number of epochs.
'dim': [int]. The dimension of the hidden layers.
'common_hid_layers_num': [int]. The number of GCN layers = 'common_hid_layers_num'+1.
'fcnn_hid_layers_num': [int]. The number of fully connected neural network layers = 'fcnn_hid_layers_num'+2.
'dropout': [float]. The probability of an element to be zeroed.
'learning_rate_SGD': [float]. Initial learning rate.
'weight_decay_SGD': [float]. L2 penalty.
'momentum': [float]. Momentum factor.
'dampening': [float]. Dampening for momentum.
'nesterov': [bool]. Enables Nesterov momentum.
'early_stopping_patience': [int]. Early stopping epochs.
'clip_grad_max_norm': [float]. Clips gradient norm of an iterable of parameters.
#'LambdaLR_scheduler_coefficient': [float]. The coefficent of the LambdaLR scheduler fucntion:  lr(epoch) = [LambdaLR_scheduler_coefficient] ^ epoch_n × learning_rate_SGD.
'print_loss_epoch_step': [int]. Print the loss value at every 'print_epoch_step' epoch.

"""
GCN_paras = {
    'epoch_n': 3000,
    'dim': 80,
    'common_hid_layers_num': 1,
    'fcnn_hid_layers_num': 1,
    'dropout': 0,
    'learning_rate_SGD': 2e-1,
    'weight_decay_SGD': 3e-4,
    'momentum': 0.9,
    'dampening': 0,
    'nesterov': True,
    'early_stopping_patience': 20,
    'clip_grad_max_norm': 1,
    'print_loss_epoch_step': 20,
}
"""## run STdGCN

Parameters 'load_test_groundtruth': [bool]. Select whether you need to upload the ground
truth file (ST_ground_truth.tsv) of the spatial transcriptomics data to track the
performance of STdGCN. 'use_marker_genes': [bool]. Select whether you need the gene
selection process before running STdGCN. Otherwise use common genes from single cell and
spatial transcriptomics data. 'external_genes': [bool]. When "use_marker_genes"=True,
you can upload your specified gene list (marker_genes.tsv) to run STdGCN.
'generate_new_pseudo_spots': [bool]. STdGCN will save the simulated pseudo-spots to
"pseudo_ST.pkl". If you want to run multiple deconvolutions with the same single cell
reference data,                     you don't need to simulate new pseudo-spots and set
'generate_new_pseudo_spots'=False. When 'generate_new_pseudo_spots'=False, you need to
pre-move the "pseudo_ST.pkl"                     to the 'output_path' so that STdGCN can
directly load the pre-simulated pseudo-spots. 'fraction_pie_plot': [bool]. Select
whether you need to draw the pie plot of the predicted results. Based on our experience,
we do not recommend to draw the pie plot when the predicted                     spot
number is very large. For 1,000 spots, the plotting time is less than 2 minutes; for
2,000 spots, the plotting time is about 10 minutes; for 3,000 spots, it takes about 30
minutes. 'cell_type_distribution_plot': [bool]. Select whether you need to draw the
scatter plot of the predicted results for each cell type. 'n_jobs': [int]. Set the
number of threads used for intraop parallelism on CPU. 'n_jobs=-1' represents using all
CPUs. 'GCN_device': ['GPU', 'CPU']. Select the device used to run GCN networks.

"""
dataset = CellTypeDeconvoDataset(data_dir=paths['datadir'], data_id=paths['dataset'])
data = dataset.load_data(
    transform=Compose([
        SetConfig({
            "feature_channel": [None, "spatial"],
            "feature_channel_type": ["X", "obsm"],
            "label_channel": "cell_type_portion",
        })
    ]), cache=False)

# sc_path = paths['sc_path']
# ST_path = paths['ST_path']
# output_path = paths['output_path']
inputs, y = data.get_data(split_name="test", return_type="torch")
x, spatial = inputs
ref_count = data.get_feature(split_name="ref", return_type="numpy")
ref_annot = data.get_feature(split_name="ref", return_type="numpy", channel="cellType", channel_type="obs")
# sc_adata = sc.read_csv(sc_path+"/sc_data.tsv", delimiter='\t')
# sc_label = pd.read_table(sc_path+"/sc_label.tsv", sep = '\t', header = 0, index_col = 0, encoding = "utf-8")
# sc_label.columns = ['cell_type']
# sc_adata.obs['cell_type'] = sc_label['cell_type'].values
sc_adata = ad.AnnData(X=ref_count)
sc_adata.obs['cell_type'] = ref_annot
cell_type_num = len(sc_adata.obs['cell_type'].unique())
cell_types = sc_adata.obs['cell_type'].unique()

word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

celltype_idx = [word_to_idx_celltype[w] for w in sc_adata.obs['cell_type']]
sc_adata.obs['cell_type_idx'] = celltype_idx
sc_adata.obs['cell_type'].value_counts()

selected_genes, cell_type_marker_genes = find_marker_genes(
    sc_adata, preprocess=find_marker_genes_paras['preprocess'],
    highly_variable_genes=find_marker_genes_paras['highly_variable_genes'],
    PCA_components=find_marker_genes_paras['PCA_components'],
    filter_wilcoxon_marker_genes=find_marker_genes_paras['filter_wilcoxon_marker_genes'],
    marker_gene_method=find_marker_genes_paras['marker_gene_method'],
    pvals_adj_threshold=find_marker_genes_paras['pvals_adj_threshold'],
    log_fold_change_threshold=find_marker_genes_paras['log_fold_change_threshold'],
    min_within_group_fraction_threshold=find_marker_genes_paras['min_within_group_fraction_threshold'],
    max_between_group_fraction_threshold=find_marker_genes_paras['max_between_group_fraction_threshold'],
    top_gene_per_type=find_marker_genes_paras['top_gene_per_type'])
# with open(output_path+"/marker_genes.tsv", 'w') as f:
#     for gene in selected_genes:
#         f.write(str(gene) + '\n')

print("{} genes have been selected as marker genes.".format(len(selected_genes)))
n_jobs = -1
pseudo_adata = pseudo_spot_generation(sc_adata, idx_to_word_celltype, spot_num=pseudo_spot_simulation_paras['spot_num'],
                                      min_cell_number_in_spot=pseudo_spot_simulation_paras['min_cell_num_in_spot'],
                                      max_cell_number_in_spot=pseudo_spot_simulation_paras['max_cell_num_in_spot'],
                                      max_cell_types_in_spot=pseudo_spot_simulation_paras['max_cell_types_in_spot'],
                                      generation_method=pseudo_spot_simulation_paras['generation_method'],
                                      n_jobs=n_jobs)
# data_file = open(output_path+'/pseudo_ST.pkl','wb')
# pickle.dump(pseudo_adata, data_file)
# data_file.close()

# ST_adata = sc.read_csv(ST_path+"/ST_data.tsv", delimiter='\t')
# ST_coor = pd.read_table(ST_path+"/coordinates.csv", sep = ',', header = 0, index_col = 0, encoding = "utf-8")
ST_adata = ad.AnnData(X=x)
ST_adata.obs['coor_X'] = spatial['x']
ST_adata.obs['coor_Y'] = spatial['y']
ST_groundtruth = y
assert y.columns == cell_types
for i in cell_types:
    ST_adata.obs[i] = ST_groundtruth[i]

ST_genes = ST_adata.var.index.values
pseudo_genes = pseudo_adata.var.index.values
common_genes = set(ST_genes).intersection(set(pseudo_genes))
ST_adata_filter = ST_adata[:, list(common_genes)]
pseudo_adata_filter = pseudo_adata[:, list(common_genes)]
pseudo_adata_norm = ST_preprocess(
    pseudo_adata_filter,
    normalize=data_normalization_paras['normalize'],
    log=data_normalization_paras['log'],
    scale=data_normalization_paras['scale'],
)[:, selected_genes]

ST_adata_filter_norm = ST_preprocess(
    ST_adata_filter,
    normalize=data_normalization_paras['normalize'],
    log=data_normalization_paras['log'],
    scale=data_normalization_paras['scale'],
)[:, selected_genes]


def update_anndata_obs(target_adata: ad.AnnData, source_adata: Optional[ad.AnnData] = None,
                       cell_types_list: Optional[List[str]] = None) -> None:
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
    if cell_types_list:
        for ct_col in cell_types_list:
            try:
                if source_adata is not None and ct_col in source_adata.obs:
                    target_adata.obs[ct_col] = source_adata.obs[ct_col]
                    print(f"Info: Column '{ct_col}' copied from source_adata.")
                else:
                    raise KeyError(f"'{ct_col}' not in source_adata or source_adata is None.")
            except (KeyError, AttributeError) as e_source_ct:
                print(f"Warning: Could not get '{ct_col}' from source_adata ({e_source_ct}). Initializing with zeros.")
                target_adata.obs[ct_col] = [0] * num_obs_target
            except Exception as e_assign:
                print(f"Error: Failed to assign '{ct_col}'. Initializing with zeros. Details: {e_assign}")
                target_adata.obs[ct_col] = [0] * num_obs_target
    else:
        print("Info: cell_types_list is None or empty. Skipping cell type column processing.")
    if cell_types_list:
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


update_anndata_obs(ST_adata_filter_norm, ST_adata_filter, cell_types)

pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types] > 0).sum(axis=1)
# results =  run_STdGCN(
#                       integration_for_adj_paras = integration_for_adj_paras,
#                       inter_exp_adj_paras = inter_exp_adj_paras,
#                       spatial_adj_paras = spatial_adj_paras,
#                       real_intra_exp_adj_paras = real_intra_exp_adj_paras,
#                       pseudo_intra_exp_adj_paras = pseudo_intra_exp_adj_paras,
#                       integration_for_feature_paras = integration_for_feature_paras,
#                       GCN_paras = GCN_paras,
#                       n_jobs = n_jobs,
#                       GCN_device = 'GPU'
#                      )

from typing import Any, List, Mapping, Optional, Tuple, Union


class stdGCNWrapper(BaseRegressionMethod):

    def __init__(self, integration_for_adj_paras, inter_exp_adj_paras, spatial_adj_paras, real_intra_exp_adj_paras,
                 pseudo_intra_exp_adj_paras, integration_for_feature_paras, GCN_paras, GCN_device='CPU', n_jobs=-1):
        self.integration_for_adj_paras = integration_for_adj_paras
        self.inter_exp_adj_paras = inter_exp_adj_paras
        self.spatial_adj_paras = spatial_adj_paras
        self.real_intra_exp_adj_paras = real_intra_exp_adj_paras
        self.ppseudo_intra_exp_adj_paras = pseudo_intra_exp_adj_paras
        self.integration_for_feature_paras = integration_for_feature_paras
        self.GCN_paras = GCN_paras
        self.GCN_device = GCN_device
        self.n_jobs = n_jobs

    @staticmethod
    def preprocessing_pipeline(log_level: LogLevel = "INFO"):
        pass

    def _init_model():
        pass

    def score(self, x, y, *, score_func: Optional[Union[str, Mapping[Any, float]]] = None, return_pred: bool = False,
              valid_idx=None, test_idx=None) -> Union[float, Tuple[float, Any]]:
        pass

    def fit_score(self, x, y, *, score_func: Optional[Union[str, Mapping[Any,
                                                                         float]]] = None, return_pred: bool = False,
                  valid_idx=None, test_idx=None, **fit_kwargs) -> Union[float, Tuple[float, Any]]:
        """Shortcut for fitting data using the input feature and return eval.

        Note
        ----
        Only work for models where the fitting does not require labeled data, i.e. unsupervised methods.

        """
        self.fit(x, **fit_kwargs)
        return self.score(x, y, score_func=score_func, return_pred=return_pred, valid_idx=valid_idx, test_idx=test_idx)

    def fit(
        self,
        inputs: Tuple[np.ndarray, np.ndarray],
        y: Optional[Any] = None,
    ):
        x, spatial = inputs
        ST_integration = data_integration(
            ST_adata_filter_norm, pseudo_adata_norm,
            batch_removal_method=self.integration_for_adj_paras['batch_removal_method'],
            dim=min(self.integration_for_adj_paras['dim'], int(ST_adata_filter_norm.shape[1] / 2)),
            dimensionality_reduction_method=self.integration_for_adj_paras['dimensionality_reduction_method'],
            scale=self.integration_for_adj_paras['scale'], cpu_num=self.n_jobs, AE_device=self.GCN_device)

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
            ST_adata_filter_norm,
            find_neighbor_method=self.real_intra_exp_adj_paras['find_neighbor_method'],
            dist_method=self.real_intra_exp_adj_paras['dist_method'],
            PCA_dimensionality_reduction=self.real_intra_exp_adj_paras['PCA_dimensionality_reduction'],
            corr_dist_neighbors=self.real_intra_exp_adj_paras['corr_dist_neighbors'],
        )

        A_pseudo_intra_exp = intra_exp_adj(
            pseudo_adata_norm,
            find_neighbor_method=self.pseudo_intra_exp_adj_paras['find_neighbor_method'],
            dist_method=self.pseudo_intra_exp_adj_paras['dist_method'],
            PCA_dimensionality_reduction=self.pseudo_intra_exp_adj_paras['PCA_dimensionality_reduction'],
            corr_dist_neighbors=self.pseudo_intra_exp_adj_paras['corr_dist_neighbors'],
        )

        real_num = ST_adata_filter.shape[0]
        pseudo_num = pseudo_adata.shape[0]

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

        ST_integration_batch_removed = data_integration(
            ST_adata_filter_norm, pseudo_adata_norm,
            batch_removal_method=self.integration_for_feature_paras['batch_removal_method'],
            dim=min(int(ST_adata_filter_norm.shape[1] * 1 / 2), self.integration_for_feature_paras['dim']),
            dimensionality_reduction_method=self.integration_for_feature_paras['dimensionality_reduction_method'],
            scale=self.integration_for_feature_paras['scale'], cpu_num=self.n_jobs, AE_device=self.GCN_device)
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

        train_valid_len = pseudo_adata.shape[0]
        test_len = ST_adata_filter.shape[0]

        table1 = ST_adata_filter_norm.obs.copy()
        label1 = table1[pseudo_adata.obs.iloc[:, :-1].columns].append(pseudo_adata.obs.iloc[:, :-1])
        label1 = torch.tensor(label1.values)

        adjs = [adj_exp.float(), adj_sp.float()]

        output1, loss, trained_model = conGCN_train(
            model=model, train_valid_len=train_valid_len, train_valid_ratio=0.9, test_len=test_len, feature=feature,
            adjs=adjs, label=label1, epoch_n=epoch_n, loss_fn=loss_fn1, optimizer=optimizer, scheduler=scheduler,
            early_stopping_patience=early_stopping_patience, clip_grad_max_norm=clip_grad_max_norm,
            print_epoch_step=print_epoch_step, cpu_num=cpu_num, GCN_device=self.GCN_device)

        loss_table = pd.DataFrame(loss, columns=['train', 'valid', 'test'])

        predict_table = pd.DataFrame(
            np.exp(output1[:test_len].detach().numpy()).tolist(), index=ST_adata_filter_norm.obs.index,
            columns=pseudo_adata_norm.obs.columns[:-2])
        # predict_table.to_csv(output_path+'/predict_result.csv', index=True, header=True)

        # torch.save(trained_model, output_path+'/model_parameters')

        pred_use = np.round_(output1.exp().detach()[:test_len], decimals=4)
        cell_type_list = cell_types
        coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]

        # ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].detach().numpy())

        # torch.cuda.empty_cache()

        # return ST_adata_filter_norm
        # self.predict_result=np.exp(output1[:test_len].detach().numpy())
        self.all_predict_result = np.exp(output1.detach().numpy())

    def predict(self, x: Optional[Any] = None):
        return self.all_predict_result


stdgcnwrapper = stdGCNWrapper(integration_for_adj_paras=integration_for_adj_paras,
                              inter_exp_adj_paras=inter_exp_adj_paras, spatial_adj_paras=spatial_adj_paras,
                              real_intra_exp_adj_paras=real_intra_exp_adj_paras,
                              pseudo_intra_exp_adj_paras=pseudo_intra_exp_adj_paras,
                              integration_for_feature_paras=integration_for_feature_paras, GCN_paras=GCN_paras,
                              n_jobs=n_jobs, GCN_device='GPU')
stdgcnwrapper.fit()
results = stdgcnwrapper.predict()
# results.write_h5ad(paths['output_path']+'/results.h5ad')
