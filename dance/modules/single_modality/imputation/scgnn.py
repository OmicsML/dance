"""Reimplementation of scGNN.

Extended from https://github.com/juexinwang/scGNN

Reference
----------
Wang, Juexin, et al. "scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses."
Nature communications 12.1 (2021): 1-11.

"""

import os
import pickle as pkl
import sys
import time
from pathlib import Path

import anndata as ad
import igraph
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.cluster import (OPTICS, AffinityPropagation, AgglomerativeClustering, Birch, KMeans, MeanShift,
                             SpectralClustering)
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dance.transforms.graph_construct import scGNNgenerateAdj


class ScDataset(Dataset):

    def __init__(self, data=None, transform=None):
        """
        Args:
            data : sparse matrix.
            transform (callable, optional):
        """
        # Now lines are cells, and cols are genes
        # self.features = data.transpose()
        self.features = data
        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class ScDatasetInter(Dataset):

    def __init__(self, features, transform=None):
        """
        Internal scData
        Args:
            construct dataset from features
        """
        self.features = features
        # Now lines are cells, and cols are genes
        # self.features = self.features.transpose()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class AE(nn.Module):
    """Autoencoder for dimensional reduction."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class VAE(nn.Module):
    """Variational Autoencoder for dimensional reduction."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO
        self.dropout = dropout
        # self.dropout = Parameter(torch.FloatTensor(dropout))
        self.act = act
        # self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GCNModelVAE(nn.Module):
    """
    Parameters
    ----------

    """

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class GCNModelAE(nn.Module):
    """
    Parameters
    ----------

    """

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj, encode=False):
        z = self.encode(x, adj)
        return z, z, None


class scGNN():
    """Single-cell GNN model.

    Parameters
    ----------
    adata :
        total data in anndata format
    genelist :
        list of gene names in data
    celllist :
        list of cell names in data
    params :
        model parameters
    model_type : str
        AE or VAE, specifying model architecture
    n_hidden2 : int
        specifies hidden dimension
    activation :
        gives torch activation function for model
    dropout : float
        probability of weight dropout during training
    GAEepochs : int
        number of epochs for GAE
    Regu_epochs : int
        number of epochs for main regulizing model in cycle
    EM_epochs : int
        number of Expectation Maximation cycles
    cluster_epochs : int
        number of epochs for cluster encoders
    batch_size : int
        number of samples in minibatch
    debugMode : str
        alternative training with debug

    """

    def __init__(self, adata, genelist, celllist, params, model_type='AE', n_hidden2=32, activation=F.relu, dropout=0.1,
                 GAEepochs=200, Regu_epochs=500, EM_epochs=200, cluster_epochs=200, batch_size=128,
                 debugMode='noDebug'):  # , lr = 1e-3, weight_decay = .9):
        self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.prj_path = Path(
            __file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve().parent.resolve()
        self.save_path = self.prj_path / 'example' / 'single_modality' / 'imputation' / 'pretrained' / params.train_dataset / 'models'
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = params.device
        self.GAEepochs = GAEepochs
        self.Regu_epochs = Regu_epochs
        self.EM_epochs = EM_epochs
        self.cluster_epochs = cluster_epochs
        self.debugMode = debugMode
        # self.lr = lr
        self.batch_size = batch_size
        # self.n_hidden1 = n_hidden1
        # self.n_hidden2 = n_hidden2
        self.activation = activation
        self.adata = adata
        self.num_cells = self.adata.shape[0]
        self.num_genes = self.adata.shape[1]
        self.genelist = genelist
        self.celllist = celllist
        # self.train_data = train_data
        # self.test_data = self.dl_params.test_data.toarray()
        self.model_type = model_type
        # self.train_ids = torch.Tensor(self.train_ids.toarray())
        # self.test_ids = torch.Tensor(self.test_ids.toarray())
        # if self.model_type == 'AE':
        #     self.model = AE(dim=self.num_genes).to(self.device)
        # else:
        #     self.model = VAE(dim=self.num_genes).to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # weight_decay=self.weight_decay)
        self.kwargs = {'num_workers': 0, 'pin_memory': False} if params.cuda else {}

    def get_enum(self, reduction):
        """ Get enumeration helper function
        Parameters
        ----------
        reduction : str
            none, mean, elementwise_mean, sum
        Returns
        ----------
        ret : int
        """
        if reduction == 'none':
            ret = 0
        elif reduction == 'mean':
            ret = 1
        elif reduction == 'elementwise_mean':
            print("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
            ret = 1
        elif reduction == 'sum':
            ret = 2
        else:
            ret = -1  # remove once JIT exceptions support control flow
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        return ret

    def generateCelltypeRegu(self, listResult):
        """generateCelltypeRegu.

        Parameters
        ----------
        listResult :
            list of cell types
        Returns
        ----------
        celltypesample :
            matrix of cell types

        """
        celltypesample = np.zeros((len(listResult), len(listResult)))
        tdict = {}
        count = 0
        for item in listResult:
            if item in tdict:
                tlist = tdict[item]
            else:
                tlist = []
            tlist.append(count)
            tdict[item] = tlist
            count += 1

        for key in sorted(tdict):
            tlist = tdict[key]
            for item1 in tlist:
                for item2 in tlist:
                    celltypesample[item1, item2] = 1.0

        return celltypesample

    def sample_mask(self, idx, l):
        """Create mask.

        Parameters
        ----------
        idx :
            boolean matrix for coordinates
        l :
            size of mask
        Returns
        ----------
        mask :
            Boolean matrix for mask

        """
        mask = np.zeros(l)
        mask[idx] = 1
        mask = np.array(mask, dtype=np.bool)
        return mask

    def legacy_get_string(self, size_average, reduce, emit_warning=True):
        warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True

        if size_average and reduce:
            ret = 'mean'
        elif reduce:
            ret = 'sum'
        else:
            ret = 'none'
        if emit_warning:
            print(warning.format(ret))
        return ret

    def vallina_mse_loss_function(self, input, target, size_average=None, reduce=None, reduction='mean'):
        r"""vallina_mse_loss_function(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
        Original: Measures the element-wise mean squared error.
        See :revised from pytorch class:`~torch.nn.MSELoss` for details.
        """
        if not (target.size() == input.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                  "This will likely lead to incorrect results due to broadcasting. "
                  "Please ensure they have the same size.".format(target.size(), input.size()))
        if size_average is not None or reduce is not None:
            reduction = self.legacy_get_string(size_average, reduce)
        # Now it use regulariz type to distinguish, it can be imporved later
        # Original, for not require grads, using c++ version
        # However, it has bugs there, different number of cpu cause different results because of MKL parallel library
        # Not known yet whether GPU has same problem.
        # Solution 1: set same number of cpu when running, it works for reproduce everything but not applicable for other users
        # https://pytorch.org/docs/stable/torch.html#torch.set_num_threads
        # https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
        # Solution 2: not use C++ codes, as we did here.
        # https://github.com/pytorch/pytorch/issues/8710

        if target.requires_grad:
            ret = (input - target)**2
            # 0.001 to reduce float loss
            # ret = (0.001*input - 0.001*target) ** 2
            if reduction != 'none':
                ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        else:
            expanded_input, expanded_target = torch.broadcast_tensors(input, target)
            ret = F.mse_loss(expanded_input, expanded_target, reduction=reduction)

        # ret = (input - target) ** 2
        # # 0.001 to reduce float loss
        # # ret = (0.001*input - 0.001*target) ** 2
        # if reduction != 'none':
        #     ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret

    def regulation_mse_loss_function(self, input, target, regulationMatrix, size_average=None, reduce=None,
                                     reduction='mean'):
        r"""regulation_mse_loss_function(input, target, regulationMatrix, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor
        Measures the element-wise mean squared error for regulation input, now only support LTMG.
        See :revised from pytorch class:`~torch.nn.MSELoss` for details.
        """
        if not (target.size() == input.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                  "This will likely lead to incorrect results due to broadcasting. "
                  "Please ensure they have the same size.".format(target.size(), input.size()))
        if size_average is not None or reduce is not None:
            reduction = self.legacy_get_string(size_average, reduce)
        # Now it use regulariz type to distinguish, it can be imporved later
        ret = (input - target)**2
        # ret = (0.001*input - 0.001*target) ** 2
        ret = torch.mul(ret, regulationMatrix)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret

    def graph_mse_loss_function(self, input, target, graphregu, size_average=None, reduce=None, reduction='mean'):
        r"""graph_mse_loss_function(input, target, adj, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor
        Measures the element-wise mean squared error in graph regularizor.
        See:revised from pytorch class:`~torch.nn.MSELoss` for details.
        """
        if not (target.size() == input.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                  "This will likely lead to incorrect results due to broadcasting. "
                  "Please ensure they have the same size.".format(target.size(), input.size()))
        if size_average is not None or reduce is not None:
            reduction = self.legacy_get_string(size_average, reduce)
        # Now it use regulariz type to distinguish, it can be imporved later
        ret = (input - target)**2
        # ret = (0.001*input - 0.001*target) ** 2
        # if graphregu != None:
        # print(graphregu.type())
        # print(ret.type())
        ret = torch.matmul(graphregu, ret)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret

    def regulation01_mse_loss_function(self, input, target, regulationMatrix, size_average=None, reduce=None,
                                       reduction='mean'):
        r"""regulation_mse_loss_function(input, target, regulationMatrix, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor
        Measures the element-wise mean squared error for regulation input, now only support LTMG.
        See :revised from pytorch class:`~torch.nn.MSELoss` for details.
        """
        if not (target.size() == input.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                  "This will likely lead to incorrect results due to broadcasting. "
                  "Please ensure they have the same size.".format(target.size(), input.size()))
        if size_average is not None or reduce is not None:
            reduction = self.legacy_get_string(size_average, reduce)
        # Now it use regulariz type to distinguish, it can be imporved later
        ret = (input - target)**2
        # ret = (0.001*input - 0.001*target) ** 2
        regulationMatrix[regulationMatrix > 0] = 1
        ret = torch.mul(ret, regulationMatrix)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret

    def loss_function_graph(self, recon_x, x, mu, logvar, graphregu=None, gammaPara=1.0, regulationMatrix=None,
                            regularizer_type='noregu', reguPara=0.001, modelusage='AE', reduction='sum'):
        """Regularized by the graph information Reconstruction + KL divergence losses
        summed over all elements and batch."""
        # Original
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # Graph
        target = x
        if regularizer_type == 'Graph' or regularizer_type == 'LTMG' or regularizer_type == 'LTMG01':
            target.requires_grad = True
        else:
            target = x.detach()

        # Euclidean
        # BCE = gammaPara * vallina_mse_loss_function(recon_x, target, reduction='sum')
        BCE = gammaPara * \
              self.vallina_mse_loss_function(recon_x, target, reduction=reduction)
        if regularizer_type == 'noregu':
            loss = BCE
        elif regularizer_type == 'LTMG':
            loss = BCE + reguPara * \
                   self.regulation_mse_loss_function(
                       recon_x, target, regulationMatrix, reduction=reduction)
        elif regularizer_type == 'LTMG01':
            loss = BCE + reguPara * \
                   self.regulation01_mse_loss_function(
                       recon_x, target, regulationMatrix, reduction=reduction)
        elif regularizer_type == 'Graph':
            loss = BCE + reguPara * \
                   self.graph_mse_loss_function(
                       recon_x, target, graphregu=graphregu, reduction=reduction)
        elif regularizer_type == 'GraphR':
            loss = BCE + reguPara * \
                   self.graph_mse_loss_function(
                       recon_x, target, graphregu=1 - graphregu, reduction=reduction)
        elif regularizer_type == 'LTMG-Graph':
            loss = BCE + reguPara * self.regulation_mse_loss_function(recon_x, target, regulationMatrix,
                                                                      reduction=reduction) + \
                   reguPara * \
                   self.graph_mse_loss_function(
                       recon_x, target, graphregu=graphregu, reduction=reduction)
        elif regularizer_type == 'LTMG-GraphR':
            loss = BCE + reguPara * self.regulation_mse_loss_function(recon_x, target, regulationMatrix,
                                                                      reduction=reduction) + \
                   reguPara * \
                   self.graph_mse_loss_function(
                       recon_x, target, graphregu=1 - graphregu, reduction=reduction)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if modelusage == 'VAE':
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss + KLD

        return loss

    def loss_function_graph_celltype(self, recon_x, x, mu, logvar, graphregu=None, celltyperegu=None, gammaPara=1.0,
                                     regulationMatrix=None, regularizer_type='noregu', reguPara=0.001,
                                     reguParaCelltype=0.001, modelusage='AE', reduction='sum'):
        """Regularized by the graph information Reconstruction + KL divergence losses
        summed over all elements and batch."""
        # Original
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # Graph
        target = x
        # if regularizer_type == 'Graph' or regularizer_type == 'LTMG' or regularizer_type == 'LTMG01' or regularizer_type == 'Celltype':
        #    target.requires_grad = True
        # Euclidean
        # BCE = gammaPara * vallina_mse_loss_function(recon_x, target, reduction='sum')
        BCE = gammaPara * \
              self.vallina_mse_loss_function(recon_x, target, reduction=reduction)
        if regularizer_type == 'noregu':
            loss = BCE
        elif regularizer_type == 'LTMG':
            loss = BCE + reguPara * \
                   self.regulation_mse_loss_function(
                       recon_x, target, regulationMatrix, reduction=reduction)
        elif regularizer_type == 'LTMG01':
            loss = BCE + reguPara * \
                   self.regulation01_mse_loss_function(
                       recon_x, target, regulationMatrix, reduction=reduction)
        elif regularizer_type == 'Graph':
            loss = BCE + reguPara * \
                   self.graph_mse_loss_function(
                       recon_x, target, graphregu=graphregu, reduction=reduction)
        elif regularizer_type == 'Celltype':
            loss = BCE + reguPara * self.graph_mse_loss_function(recon_x, target, graphregu=graphregu,
                                                                 reduction=reduction) + \
                   reguParaCelltype * \
                   self.graph_mse_loss_function(
                       recon_x, target, graphregu=celltyperegu, reduction=reduction)
        elif regularizer_type == 'CelltypeR':
            loss = BCE + (1 - gammaPara) * self.regulation01_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction) + reguPara * self.graph_mse_loss_function(
                    recon_x, target, graphregu=graphregu,
                    reduction=reduction) + reguParaCelltype * self.graph_mse_loss_function(
                        recon_x, target, graphregu=celltyperegu, reduction=reduction)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if modelusage == 'VAE':
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss + KLD

        return loss

    def gae_loss_function(self, preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        """Loss function for Graph AutoEncoder.

        Parameters
        ----------
        preds :
            prediction matrix
        labels :
            true matrix
        mu : float
            parameter in KL distance
        logvar : float
            log variance
        n_nodes : int
            number of nodes in graph
        norm : float
            adjacency norm
        pos_weight : float
            weight of positive examples in cross entropy

        Returns
        -------
        loss : float
            loss summation of cross entropy and KL distance

        """
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

        # Check if the model is simple Graph Auto-encoder
        if logvar is None:
            return cost

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        loss = cost + KLD
        return loss

    def trimClustering(self, listResult, minMemberinCluster=5, maxClusterNumber=30):
        """If the clustering numbers larger than certain number, use this function to
        trim."""
        numDict = {}
        for item in listResult:
            if not item in numDict:
                numDict[item] = 0
            else:
                numDict[item] = numDict[item] + 1

        size = len(set(listResult))
        changeDict = {}
        for item in range(size):
            if numDict[item] < minMemberinCluster or item >= maxClusterNumber:
                changeDict[item] = ''

        count = 0
        for item in listResult:
            if item in changeDict:
                listResult[count] = maxClusterNumber
            count += 1

        return listResult

    def generateLouvainCluster(self, edgeList):
        """Louvain Clustering using igraph."""
        Gtmp = nx.Graph()
        Gtmp.add_weighted_edges_from(edgeList)
        W = nx.adjacency_matrix(Gtmp)
        W = W.todense()
        graph = igraph.Graph.Weighted_Adjacency(W.tolist(), mode=igraph.ADJ_UNDIRECTED, attr="weight", loops=False)
        louvain_partition = graph.community_multilevel(weights=graph.es['weight'], return_levels=False)
        size = len(louvain_partition)
        hdict = {}
        count = 0
        for i in range(size):
            tlist = louvain_partition[i]
            for j in range(len(tlist)):
                hdict[tlist[j]] = i
                count += 1

        listResult = []
        for i in range(count):
            listResult.append(hdict[i])

        return listResult, size

    def sparse_to_tuple(self, sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape

    def mask_test_edges(self, adj):
        """Function to build test set with 10% positive links.

        Parameters
        ----------
        adj :
            adjacency matrix

        """
        #
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
        # TODO: Clean up.

        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = self.sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = self.sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] / 10.))
        num_val = int(np.floor(edges.shape[0] / 20.))

        all_edge_idx = np.arange(edges.shape[0])
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            if ~ismember([idx_i, idx_j], edges_all) and ~ismember([idx_j, idx_i], edges_all):
                val_edges_false.append([idx_i, idx_j])
            else:
                # Debug
                print(str(idx_i) + " " + str(idx_j))
            # Original:
            # val_edges_false.append([idx_i, idx_j])

        # TODO: temporary disable for ismember function may require huge memory.
        # assert ~ismember(test_edges_false, edges_all)
        # assert ~ismember(val_edges_false, edges_all)
        # assert ~ismember(val_edges, train_edges)
        # assert ~ismember(test_edges, train_edges)
        # assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        # NOTE: these edge lists only contain single direction of edge!
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        # sparse_mx = sparse_mx.tocoo().astype(np.float64)
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        # return torch.sparse.DoubleTensor(indices, values, shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        # return sparse_to_tuple(adj_normalized)
        return self.sparse_mx_to_torch_sparse_tensor(adj_normalized)

    def get_roc_score(self, emb, adj_orig, edges_pos, edges_neg):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def GAEembedding(self, z, adj, params):
        """GAE embedding for clustering.

        Parameters
        ----------
        z :
            input parameters
        adj :
            adjacency matrix of cell graph
        params :
            model parameters

        Returns
        -------
        hidden_emb :
            Embedding from graph

        """
        # featrues from z
        # Louvain
        features = z
        # features = torch.DoubleTensor(features)
        features = torch.FloatTensor(features)

        # Old implementation
        # adj, features, y_test, tx, ty, test_maks, true_labels = load_data(params.dataset_str)

        n_nodes, feat_dim = features.shape

        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = self.mask_test_edges(adj)
        adj = adj_train

        # Some preprocessing
        adj_norm = self.preprocess_graph(adj)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)
        # adj_label = torch.DoubleTensor(adj_label.toarray())
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        if params.GAEmodel == 'gcn_vae':
            model = GCNModelVAE(feat_dim, params.GAEhidden1, params.GAEhidden2, params.GAEdropout)
        else:
            model = GCNModelAE(feat_dim, params.GAEhidden1, params.GAEhidden2, params.GAEdropout)
        if params.precisionModel == 'Double':
            model = model.double()
        optimizer = optim.Adam(model.parameters(), lr=params.GAElr)

        for epoch in tqdm(range(params.GAEepochs)):
            t = time.time()
            # mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption before training: '+str(mem))
            model.train()
            optimizer.zero_grad()
            z, mu, logvar = model(features, adj_norm)

            loss = self.gae_loss_function(preds=model.dc(z), labels=adj_label, mu=mu, logvar=logvar, n_nodes=n_nodes,
                                          norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy()

            # roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            ap_curr = 0

            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, val_ap={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, ap_curr,
                time.time() - t))

        tqdm.write("Optimization Finished!")

        roc_score, ap_score = self.get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
        tqdm.write('Test ROC score: ' + str(roc_score))
        tqdm.write('Test AP score: ' + str(ap_score))

        return hidden_emb

    def train(
        self,
        model,
        optimizer,
        epoch,
        params,
        train_loader,
        regulationMatrix=None,
        EMFlag=False,
        taskType='celltype',
        sparseImputation='nonsparse',
    ):
        """Training loop.

        Parameters
        ----------
        model :
            input model to train
        optimizer :
            input optimizer for model training
        epoch : int
            epoch number
        params :
            model parameters
        train_loader :
            input data class
        regulationMatrix
            LTMG regulation matrix
        EMFlag :
            EMFlag indicates whether in EM processes.
            If in EM, use regulized-type parsed from program entrance,
            Otherwise, noregu
        taskType : str
            celltype or imputation for different model training
        sparseImputation : str
            whether input is sparse format

        Returns
        ----------
        recon_batch_all :
            reconstruction of batch data
        data_all :
            return input data from train_loader
        z_all :
            hidden representation of data from model

        """
        model.train()
        train_loss = 0
        for batch_idx, (data, dataindex) in enumerate(train_loader):
            data = data.to(self.device)
            if params.precisionModel == 'Double':
                data = data.type(torch.DoubleTensor)
            elif params.precisionModel == 'Float':
                data = data.type(torch.FloatTensor)
            data = data.to(self.device)
            if not params.regulized_type == 'noregu':
                regulationMatrixBatch = regulationMatrix[dataindex, :]
                regulationMatrixBatch = regulationMatrixBatch.to(self.device)
            else:
                regulationMatrixBatch = None
            if taskType == 'imputation':
                if sparseImputation == 'nonsparse':
                    celltypesampleBatch = self.celltypesample[dataindex, :][:, dataindex]
                    adjsampleBatch = self.adjsample[dataindex, :][:, dataindex]
                elif sparseImputation == 'sparse':
                    celltypesampleBatch = self.generateCelltypeRegu(self.listResult[dataindex])
                    celltypesampleBatch = torch.from_numpy(celltypesampleBatch)
                    if params.precisionModel == 'Float':
                        celltypesampleBatch = celltypesampleBatch.float()
                    elif params.precisionModel == 'Double':
                        celltypesampleBatch = celltypesampleBatch.type(torch.DoubleTensor)
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('celltype Mem consumption: '+str(mem))

                    adjsampleBatch = self.adj[dataindex, :][:, dataindex]
                    adjsampleBatch = sp.csr_matrix.todense(adjsampleBatch)
                    adjsampleBatch = torch.from_numpy(adjsampleBatch)
                    if params.precisionModel == 'Float':
                        adjsampleBatch = adjsampleBatch.float()
                    elif params.precisionModel == 'Double':
                        adjsampleBatch = adjsampleBatch.type(torch.DoubleTensor)
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('adj Mem consumption: '+str(mem))
            optimizer.zero_grad()
            if self.model_type == 'VAE':
                recon_batch, mu, logvar, z = model(data)
                if taskType == 'celltype':
                    if EMFlag and (not params.EMreguTag):
                        loss = self.loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar,
                                                        gammaPara=params.gammaPara,
                                                        regulationMatrix=regulationMatrixBatch,
                                                        regularizer_type='noregu', reguPara=params.alphaRegularizePara,
                                                        modelusage=self.model_type, reduction=params.reduction)
                    else:
                        loss = self.loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar,
                                                        gammaPara=params.gammaPara,
                                                        regulationMatrix=regulationMatrixBatch,
                                                        regularizer_type=params.regulized_type,
                                                        reguPara=params.alphaRegularizePara, modelusage=self.model_type,
                                                        reduction=params.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not params.EMreguTag):
                        loss = self.loss_function_graph_celltype(
                            recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch,
                            celltyperegu=celltypesampleBatch, gammaPara=params.gammaImputePara,
                            regulationMatrix=regulationMatrixBatch, regularizer_type=params.EMregulized_type,
                            reguPara=params.graphImputePara, reguParaCelltype=params.celltypeImputePara,
                            modelusage=self.model_type, reduction=params.reduction)
                    else:
                        loss = self.loss_function_graph_celltype(
                            recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch,
                            celltyperegu=celltypesampleBatch, gammaPara=params.gammaImputePara,
                            regulationMatrix=regulationMatrixBatch, regularizer_type=params.regulized_type,
                            reguPara=params.graphImputePara, reguParaCelltype=params.celltypeImputePara,
                            modelusage=self.model_type, reduction=params.reduction)

            elif self.model_type == 'AE':
                recon_batch, z = model(data)
                mu_dummy = ''
                logvar_dummy = ''
                if taskType == 'celltype':
                    if EMFlag and (not params.EMreguTag):
                        loss = self.loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy,
                                                        logvar_dummy, gammaPara=params.gammaPara,
                                                        regulationMatrix=regulationMatrixBatch,
                                                        regularizer_type='noregu', reguPara=params.alphaRegularizePara,
                                                        modelusage=self.model_type, reduction=params.reduction)
                    else:
                        loss = self.loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy,
                                                        logvar_dummy, gammaPara=params.gammaPara,
                                                        regulationMatrix=regulationMatrixBatch,
                                                        regularizer_type=params.regulized_type,
                                                        reguPara=params.alphaRegularizePara, modelusage=self.model_type,
                                                        reduction=params.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not params.EMreguTag):
                        loss = self.loss_function_graph_celltype(
                            recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy,
                            graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch,
                            gammaPara=params.gammaImputePara, regulationMatrix=regulationMatrixBatch,
                            regularizer_type=params.EMregulized_type, reguPara=params.graphImputePara,
                            reguParaCelltype=params.celltypeImputePara, modelusage=self.model_type,
                            reduction=params.reduction)
                    else:
                        loss = self.loss_function_graph_celltype(
                            recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy,
                            graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch,
                            gammaPara=params.gammaImputePara, regulationMatrix=regulationMatrixBatch,
                            regularizer_type=params.regulized_type, reguPara=params.graphImputePara,
                            reguParaCelltype=params.celltypeImputePara, modelusage=self.model_type,
                            reduction=params.reduction)

            # L1 and L2 regularization
            # 0.0 for no regularization
            l1 = 0.0
            l2 = 0.0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
                l2 = l2 + p.pow(2).sum()
            # loss = loss + params.L1Para * l1 + params.L2Para * l2
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % params.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               train_loss / len(data)))

            # for batch
            if batch_idx == 0:
                recon_batch_all = recon_batch
                data_all = data
                z_all = z
            else:
                recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
                data_all = torch.cat((data_all, data), 0)
                z_all = torch.cat((z_all, z), 0)

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        return recon_batch_all, data_all, z_all

    def fit(self, train_data):
        """scGNN fit function.

        Parameters
        ----------
        train_data :
            input gene expression features

        Returns
        -------
        reconOri :
            Initial data reconstruction used for final imputation

        """
        if sp.issparse(train_data):
            train_data = train_data.toarray()
        data = train_data
        scData = ScDataset(data)
        train_loader = DataLoader(scData, batch_size=self.batch_size, shuffle=False, **self.kwargs)
        print('---TrainLoader has been successfully prepared.')
        if self.model_type == 'VAE':
            model = VAE(dim=data.shape[1]).to(self.device)
        elif self.model_type == 'AE':
            model = AE(dim=data.shape[1]).to(self.device)
        if self.params.precisionModel == 'Double':
            model = model.double()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print('---Pytorch model ready.')
        if not os.path.exists(self.params.outputDir):
            os.makedirs(self.params.outputDir)
        ptfileStart = self.params.outputDir + self.params.train_dataset + '_EMtrainingStart.pt'

        if self.params.debugMode == 'savePrune' or self.params.debugMode == 'noDebug':
            stateStart = {
                # 'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(stateStart, ptfileStart)
            for epoch in range(1, self.Regu_epochs + 1):
                recon, original, z = self.train(model, optimizer, epoch, self.params, train_loader, EMFlag=False)

            zOut = z.detach().cpu().numpy()
            print('zOut ready')
            ptstatus = model.state_dict()

            # Store reconOri for imputation
            reconOri = recon.clone()
            reconOri = reconOri.detach().cpu().numpy()

            # Step 1. Inferring celltype

            self.adj, self.edgeList = scGNNgenerateAdj(
                zOut, graphType=self.params.prunetype, para=self.params.knn_distance + ':' + str(self.params.k),
                adjTag=(self.params.useGAEembedding or self.params.useBothembedding))

        if self.debugMode == 'savePrune':
            # Add protocol=4 for serizalize object larger than 4GiB
            with open('edgeListFile', 'wb') as edgeListFile:
                pkl.dump(self.edgeList, edgeListFile, protocol=4)

            with open('adjFile', 'wb') as adjFile:
                pkl.dump(self.adj, adjFile, protocol=4)

            with open('zOutFile', 'wb') as zOutFile:
                pkl.dump(zOut, zOutFile, protocol=4)

            with open('reconFile', 'wb') as reconFile:
                pkl.dump(recon, reconFile, protocol=4)

            with open('originalFile', 'wb') as originalFile:
                pkl.dump(original, originalFile, protocol=4)

            sys.exit(0)

        if self.params.debugMode == 'loadPrune':
            with open('edgeListFile', 'rb') as edgeListFile:
                self.edgeList = pkl.load(edgeListFile)

            with open('adjFile', 'rb') as adjFile:
                self.adj = pkl.load(adjFile)

            with open('zOutFile', 'rb') as zOutFile:
                zOut = pkl.load(zOutFile)

            with open('reconFile', 'rb') as reconFile:
                recon = pkl.load(reconFile)

            with open('originalFile', 'rb') as originalFile:
                original = pkl.load(originalFile)

        if self.params.useGAEembedding or self.params.useBothembedding:
            zDiscret = zOut > np.mean(zOut, axis=0)
            zDiscret = 1.0 * zDiscret
            if self.params.useGAEembedding:
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))
                zOut = self.GAEembedding(zDiscret, self.adj, self.params)
                print("---GAE embedding finished")
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))

            elif self.params.useBothembedding:
                zEmbedding = self.GAEembedding(zDiscret, self.adj, self.params)
                zOut = np.concatenate((zOut, zEmbedding), axis=1)

        G0 = nx.Graph()
        G0.add_weighted_edges_from(self.edgeList)
        nlG0 = nx.normalized_laplacian_matrix(G0)
        # set iteration criteria for converge
        adjOld = nlG0
        # set celltype criteria for converge
        listResultOld = [1 for i in range(zOut.shape[0])]

        # Fill the zeros before EM iteration
        # need better implementation later
        # this parameter largely unhelpful
        # broken code anyways
        # if self.params.zerofillFlag:
        #     for nz_index in range(len(scData.nz_i)):
        #         # tmp = scipy.sparse.lil_matrix.todense(scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]])
        #         # tmp = np.asarray(tmp).reshape(-1)[0]
        #         tmp = scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]]
        #         reconOut[scData.nz_i[nz_index], scData.nz_j[nz_index]] = tmp
        #     recon = reconOut

        # Define resolution
        # Default: auto, otherwise use user defined resolution
        if self.params.resolution == 'auto':
            if zOut.shape[0] < 2000:
                resolution = 0.8
            else:
                resolution = 0.5
        else:
            resolution = float(self.params.resolution)

        print("---EM process starts")

        for bigepoch in range(0, self.params.EM_iteration):
            print('---Start %sth iteration.' % (bigepoch))
            # Now for both methods, we need do clustering, using clustering results to check converge
            # Clustering: Get clusters
            if self.params.clustering_method == 'Louvain':
                listResult, size = self.generateLouvainCluster(self.edgeList)
                k = len(np.unique(listResult))
                print('Louvain cluster: ' + str(k))
            elif self.params.clustering_method == 'LouvainK':
                listResult, size = self.generateLouvainCluster(self.edgeList)
                k = len(np.unique(listResult))
                print('Louvain cluster: ' + str(k))
                k = int(k * resolution) if int(k * resolution) >= 3 else 2
                clustering = KMeans(n_clusters=k, random_state=0).fit(zOut)
                # clustering = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=128).fit(zOut)
                listResult = clustering.predict(zOut)
            elif self.params.clustering_method == 'LouvainB':
                listResult, size = self.generateLouvainCluster(self.edgeList)
                k = len(np.unique(listResult))
                print('Louvain cluster: ' + str(k))
                k = int(k * resolution) if int(k * resolution) >= 3 else 2
                clustering = Birch(n_clusters=k).fit(zOut)
                listResult = clustering.predict(zOut)
            elif self.params.clustering_method == 'KMeans':
                clustering = KMeans(n_clusters=self.params.n_clusters, random_state=0).fit(zOut)
                listResult = clustering.predict(zOut)
            elif self.params.clustering_method == 'SpectralClustering':
                clustering = SpectralClustering(n_clusters=self.params.n_clusters, assign_labels="discretize",
                                                random_state=0).fit(zOut)
                listResult = clustering.labels_.tolist()
            elif self.params.clustering_method == 'AffinityPropagation':
                clustering = AffinityPropagation().fit(zOut)
                listResult = clustering.predict(zOut)
            elif self.params.clustering_method == 'AgglomerativeClustering':
                clustering = AgglomerativeClustering().fit(zOut)
                listResult = clustering.labels_.tolist()
            elif self.params.clustering_method == 'AgglomerativeClusteringK':
                clustering = AgglomerativeClustering(n_clusters=self.params.n_clusters).fit(zOut)
                listResult = clustering.labels_.tolist()
            elif self.params.clustering_method == 'Birch':
                clustering = Birch(n_clusters=self.params.n_clusters).fit(zOut)
                listResult = clustering.predict(zOut)
            elif self.params.clustering_method == 'BirchN':
                clustering = Birch(n_clusters=None).fit(zOut)
                listResult = clustering.predict(zOut)
            elif self.params.clustering_method == 'MeanShift':
                clustering = MeanShift().fit(zOut)
                listResult = clustering.predict(zOut)
            elif self.params.clustering_method == 'OPTICS':
                clustering = OPTICS(min_samples=int(self.params.k / 2),
                                    min_cluster_size=self.params.minMemberinCluster).fit(zOut)
                listResult = clustering.predict(zOut)
            else:
                print("Error: Clustering method not appropriate")
            print("---Clustering Ends")

            # If clusters more than maxclusters, then have to stop
            if len(set(listResult)) > self.params.maxClusterNumber or len(set(listResult)) <= 1:
                print("Stopping: Number of clusters is " + str(len(set(listResult))) + ".")
                # Exit
                # return None
                # Else: dealing with the number
                listResult = self.trimClustering(listResult, minMemberinCluster=self.params.minMemberinCluster,
                                                 maxClusterNumber=self.params.maxClusterNumber)

            # Debug: Calculate silhouette
            # measure_clustering_results(zOut, listResult)
            print('Total Cluster Number: ' + str(len(set(listResult))))
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

            # Graph regulizated EM AE with Cluster AE, do the additional AE
            if not self.params.quickmode:
                # Each cluster has a autoencoder, and organize them back in iteraization
                print('---Start Cluster Autoencoder.')
                clusterIndexList = []
                for i in range(len(set(listResult))):
                    clusterIndexList.append([])
                for i in range(len(listResult)):
                    assignee = listResult[i]
                    # Avoid bugs for maxClusterNumber
                    if assignee == self.params.maxClusterNumber:
                        assignee = self.params.maxClusterNumber - 1
                    clusterIndexList[assignee].append(i)

                reconNew = np.zeros((scData.features.shape[0], scData.features.shape[1]))

                # Convert to Tensor
                reconNew = torch.from_numpy(reconNew)
                if self.params.precisionModel == 'Double':
                    reconNew = reconNew.type(torch.DoubleTensor)
                elif self.params.precisionModel == 'Float':
                    reconNew = reconNew.type(torch.FloatTensor)
                reconNew = reconNew.to(self.device)

                model.load_state_dict(ptstatus)

                for clusterIndex in clusterIndexList:
                    reconUsage = recon[clusterIndex]
                    scDataInter = ScDatasetInter(reconUsage)
                    train_loader = DataLoader(scDataInter, batch_size=self.batch_size, shuffle=False, **self.kwargs)

                    for epoch in range(1, self.cluster_epochs + 1):
                        reconCluster, originalCluster, zCluster = self.train(model, optimizer, epoch, self.params,
                                                                             train_loader, EMFlag=True)
                    count = 0
                    for i in clusterIndex:
                        reconNew[i] = reconCluster[count, :]
                        count += 1
                    # empty cuda cache
                    del originalCluster
                    del zCluster
                    torch.cuda.empty_cache()

                # Update
                recon = reconNew
                ptstatus = model.state_dict()

                # Debug mem consumption
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))

            # Use new dataloader
            scDataInter = ScDatasetInter(recon.detach().to('cpu'))
            train_loader = DataLoader(scDataInter, batch_size=self.batch_size, shuffle=False, **self.kwargs)

            for epoch in range(1, self.EM_epochs + 1):
                recon, original, z = self.train(model, optimizer, epoch, self.params, train_loader, EMFlag=True)
                # recon, original, z = train(epoch, train_loader=train_loader, EMFlag=True)

            zOut = z.detach().cpu().numpy()

            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            print('---Start Prune')

            self.adj, self.edgeList = scGNNgenerateAdj(
                zOut, graphType=self.params.prunetype, para=self.params.knn_distance + ':' + str(self.params.k),
                adjTag=(self.params.useGAEembedding or self.params.useBothembedding
                        or (bigepoch == int(self.params.EM_iteration) - 1)))

            print('---Prune Finished')
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

            # Whether use GAE embedding
            if self.params.useGAEembedding or self.params.useBothembedding:
                zDiscret = zOut > np.mean(zOut, axis=0)
                zDiscret = 1.0 * zDiscret
                if self.params.useGAEembedding:
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('Mem consumption: '+str(mem))
                    zOut = self.GAEembedding(zDiscret, self.adj, self.params)
                    print("---GAE embedding finished")
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('Mem consumption: '+str(mem))
                elif self.params.useBothembedding:
                    zEmbedding = self.GAEembedding(zDiscret, self.adj, self.params)
                    zOut = np.concatenate((zOut, zEmbedding), axis=1)

            # Original save step by step
            if self.params.saveinternal:
                print('---Start save internal results')
                reconOut = recon.detach().cpu().numpy()

                # Output
                print('---Prepare save')
                # print('Save results with reconstructed shape:'+str(reconOut.shape)+' Size of gene:'+str(len(genelist))+' Size of cell:'+str(len(celllist)))
                recon_df = pd.DataFrame(np.transpose(reconOut), index=self.genelist, columns=self.celllist)
                recon_df.to_csv(self.params.outputDir + self.params.datasetName + '_' + self.params.regulized_type +
                                '_' + str(self.params.alphaRegularizePara) + '_' + str(self.params.L1Para) + '_' +
                                str(self.params.L2Para) + '_recon_' + str(bigepoch) + '.csv')
                emblist = []
                for i in range(zOut.shape[1]):
                    emblist.append('embedding' + str(i))
                embedding_df = pd.DataFrame(zOut, index=self.celllist, columns=emblist)
                embedding_df.to_csv(self.params.outputDir + self.params.datasetName + '_' + self.params.regulized_type +
                                    '_' + str(self.params.alphaRegularizePara) + '_' + str(self.params.L1Para) + '_' +
                                    str(self.params.L2Para) + '_embedding_' + str(bigepoch) + '.csv')
                graph_df = pd.DataFrame(self.edgeList, columns=["NodeA", "NodeB", "Weights"])
                graph_df.to_csv(
                    self.params.outputDir + self.params.datasetName + '_' + self.params.regulized_type + '_' +
                    str(self.params.alphaRegularizePara) + '_' + str(self.params.L1Para) + '_' +
                    str(self.params.L2Para) + '_graph_' + str(bigepoch) + '.csv', index=False)
                results_df = pd.DataFrame(listResult, index=self.celllist, columns=["Celltype"])
                results_df.to_csv(self.params.outputDir + self.params.datasetName + '_' + self.params.regulized_type +
                                  '_' + str(self.params.alphaRegularizePara) + '_' + str(self.params.L1Para) + '_' +
                                  str(self.params.L2Para) + '_results_' + str(bigepoch) + '.txt')

                print('---Save internal completed')

            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            print('---Start test converge condition')

            # Iteration usage
            # If not only use 'celltype', we have to use graph change
            # The problem is it will consume huge memory for giant graphs
            if not self.params.converge_type == 'celltype':
                Gc = nx.Graph()
                Gc.add_weighted_edges_from(self.edgeList)
                adjGc = nx.adjacency_matrix(Gc)

                # Update new adj
                adjNew = self.params.alpha * nlG0 + \
                         (1 - self.params.alpha) * adjGc / np.sum(adjGc, axis=0)

                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))
                print('---New adj ready')

                # debug
                graphChange = np.mean(abs(adjNew - adjOld))
                graphChangeThreshold = self.params.converge_graphratio * \
                                       np.mean(abs(nlG0))
                print('adjNew:{} adjOld:{} G0:{}'.format(adjNew, adjOld, nlG0))
                print('mean:{} threshold:{}'.format(graphChange, graphChangeThreshold))

                # Update
                adjOld = adjNew

            # Check similarity
            ari = adjusted_rand_score(listResultOld, listResult)

            # Debug Information of clustering results between iterations
            # print(listResultOld)
            # print(listResult)
            print('celltype similarity:' + str(ari))

            # graph criteria
            if self.params.converge_type == 'graph':
                if graphChange < graphChangeThreshold:
                    print('Converge now!')
                    break
            # celltype criteria
            elif self.params.converge_type == 'celltype':
                if ari > self.params.converge_celltyperatio:
                    print('Converge now!')
                    break
            # if both criteria are meets
            elif self.params.converge_type == 'both':
                if graphChange < graphChangeThreshold and ari > self.params.converge_celltyperatio:
                    print('Converge now!')
                    break
            # if either criteria are meets
            elif self.params.converge_type == 'either':
                if graphChange < graphChangeThreshold or ari > self.params.converge_celltyperatio:
                    print('Converge now!')
                    break

            # Update
            listResultOld = listResult
            self.listResult = listResult

            print("---" + str(bigepoch) + "th iteration in EM Finished")
            self.model = model
            self.optimizer = optimizer
            self.ptfileStart = ptfileStart
            self.zOut = zOut
            return reconOri

    def predict(self, test_data):
        """Predict function for imputation.

        Parameters
        ----------
        reconOri :
            input to be imputed

        Returns
        -------
        embedding_df :
            dataframe embedding of
        graph_df :
            dataframe of edge list cell graph
        results_df :
            dataframe of cell types
        recon_df :
            dataframe of reconstructed input, completed imputed values

        """
        print("---Starts Imputation")
        test_data = torch.Tensor(test_data).to(self.device)
        stateStart = torch.load(self.ptfileStart)
        self.model.load_state_dict(stateStart['state_dict'])
        self.optimizer.load_state_dict(stateStart['optimizer'])

        if self.model_type == 'VAE':
            recon, _, _, z = self.model(test_data)
        elif self.model_type == 'AE':
            recon, z = self.model(test_data)

        zOut = z.detach().cpu().numpy()
        reconOut = recon.detach().cpu().numpy()
        if not self.params.noPostprocessingTag:
            threshold_indices = reconOut < self.params.postThreshold
            reconOut[threshold_indices] = 0.0

        return reconOut

    def score(self, true_expr, imputed_expr, test_idx, metric, true_labels=None, n_neighbors=None, n_pcs=None,
              clu_resolution=1, targetgenes=None):
        """Evaluate the trained model.

        Parameters
        ----------
        true_expr : DataFrame
            true expression
        imputed_expr : DataFrame
            imputed expression
        metric : string
            choice of metric, either MSE or ARI
        true_labels : array optional
            provided cell labels
        n_neighbors : int optional
            number of neighbors tp cluster imputed data
        n_pcs: int optional
            number of principal components for neighbor detection
        clu_resolution : int optional
            resolution for Leiden cluster
        targetgenes: array optional
            genes to be imputed

        Returns
        -------
        acc : float
            accuracy.
        nmi : float
            normalized mutual information.
        ari : float
            adjusted Rand index.
        mse : float
            mean squared erros

        """
        if sp.issparse(true_expr):
            true_expr = true_expr.toarray()
        # if imputed_expr == None:
        #     imputed_expr = self.predict(self.reconOri)
        if (targetgenes == None):
            targetgenes = self.genelist
        # targetgenes = targetgenes.flatten()
        allowd_metrics = {"MSE", "ARI"}
        if metric not in allowd_metrics:
            raise ValueError("scoring metric %r." % allowd_metrics)
        # imputed_expr = imputed_expr.loc[:,targetgenes] # subset target genes only
        if (metric == 'MSE'):
            true_target = true_expr[test_idx, ]
            imputed_target = imputed_expr[test_idx, ]
            # true_expr = true_expr.loc[imputed_expr.index, imputed_expr.columns]
            mse_cells = pd.DataFrame(((true_target - imputed_target)**2).mean(axis=1)).dropna()
            mse_genes = pd.DataFrame(((true_target - imputed_target)**2).mean(axis=0)).dropna()
            return mse_cells, mse_genes
        elif (metric == 'ARI'):
            adata = ad.AnnData(imputed_expr)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata)
            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=clu_resolution)
            pred_labels = adata.obs['leiden']
            ari = round(adjusted_rand_score(true_labels, pred_labels), 3)
            return ari
