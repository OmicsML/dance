from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

from dance import data, logger
from dance.modules.base import BaseClassificationMethod
from dance.typing import Any, Mapping, Optional, Tuple, Union
from dance.utils.metrics import resolve_score_func


#TODO 关于eval_acc和score的部分有待核实
#TODO train valid test的score可以加到result里，就可以获得全局或者局部score
def eval_acc(true_labels, model_output, acc):
    predicted_indices = torch.argmax(model_output, dim=1)  # Shape [N], on device
    num_classes = model_output.shape[1]  # Get number of classes from model output
    if true_labels.ndim == 1:
        true_labels_squeezed = true_labels
    elif true_labels.shape[1] == 1:
        true_labels_squeezed = true_labels.squeeze(1)
    else:
        print(
            "Warning: true_labels shape suggests it might already be one-hot or multi-label. Assuming conversion needed for standard accuracy."
        )
        true_labels_squeezed = true_labels  # Or handle appropriately if known to be one-hot

    # Ensure labels are integer type for one_hot
    true_labels_long = true_labels_squeezed.long()

    true_one_hot = F.one_hot(
        true_labels_long,
        num_classes=num_classes)  # Needs labels on CPU? Check F.one_hot docs if issues arise. Usually works on device.

    accuary = acc(true_one_hot, predicted_indices)
    return accuary


def print_statistics(results, run=None):
    if run is not None:
        result = 100 * torch.tensor(results[run])
        ood_result, test_score, valid_loss = result[:, :-2], result[:, -2], result[:, -1]
        argmin = valid_loss.argmin().item()
        print(f'Run {run + 1:02d}:')
        print(f'Chosen epoch: {argmin + 1}')
        for k in range(result.shape[1] // 3):
            print(f'OOD Test {k+1} Final AUROC: {ood_result[argmin, k*3]:.2f}')
            print(f'OOD Test {k+1} Final AUPR: {ood_result[argmin, k*3+1]:.2f}')
            print(f'OOD Test {k+1} Final FPR95: {ood_result[argmin, k*3+2]:.2f}')
        print(f'IND Test Score: {test_score[argmin]:.2f}')
    else:
        result = 100 * torch.tensor(results)

        ood_te_num = result.shape[2] // 3

        best_results = []
        for r in result:
            ood_result, test_score, valid_loss = r[:, :-2], r[:, -2], r[:, -1]
            score_val = test_score[valid_loss.argmin()].item()
            ood_result_val = []
            for k in range(ood_te_num):
                auroc_val = ood_result[valid_loss.argmin(), k * 3].item()
                aupr_val = ood_result[valid_loss.argmin(), k * 3 + 1].item()
                fpr_val = ood_result[valid_loss.argmin(), k * 3 + 2].item()
                ood_result_val += [auroc_val, aupr_val, fpr_val]
            best_results.append(ood_result_val + [score_val])

        best_result = torch.tensor(best_results)

        if best_result.shape[0] == 1:
            print(f'All runs:')
            for k in range(ood_te_num):
                r = best_result[:, k * 3]
                print(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f}')
                r = best_result[:, k * 3 + 1]
                print(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f}')
                r = best_result[:, k * 3 + 2]
                print(f'OOD Test {k + 1} Final FPR: {r.mean():.2f}')
            r = best_result[:, -1]
            print(f'IND Test Score: {r.mean():.2f}')
        else:
            print(f'All runs:')
            for k in range(ood_te_num):
                r = best_result[:, k * 3]
                print(f'OOD Test {k+1} Final AUROC: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, k * 3 + 1]
                print(f'OOD Test {k+1} Final AUPR: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, k * 3 + 2]
                print(f'OOD Test {k+1} Final FPR: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, -1]
            print(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}')

        return best_result


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``

    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None
            and not (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1])
                     or np.array_equal(classes, [0]) or np.array_equal(classes, [-1]) or np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, threshould


dataset_prefix = '_processed'


class NCDataset:

    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def build_graph(adata, radius=None, knears=None, distance_metrics='l2'):
    """
    based on https://github.com/hannshu/st_datasets/blob/master/utils/preprocess.py
    """
    if (isinstance(adata.X, np.ndarray)):
        coor = pd.DataFrame(adata.X)
    else:
        coor = pd.DataFrame(adata.X.todense())

    if (radius):
        nbrs = NearestNeighbors(radius=radius, metric=distance_metrics).fit(coor)
        _, indices = nbrs.radius_neighbors(coor, return_distance=True)
    else:
        nbrs = NearestNeighbors(n_neighbors=knears + 1, metric=distance_metrics).fit(coor)
        _, indices = nbrs.kneighbors(coor)

    edge_list = np.array([[i, j] for i, sublist in enumerate(indices) for j in sublist])
    return edge_list


def load_dataset_fixed(ref_adata, ref_adata_name, ignore_first=False, ood=False, knn_num=5):
    dataset = NCDataset(ref_adata_name)
    # encoding label to id
    y = ref_adata.obsm['cell_type'].copy()
    X = ref_adata.X.copy()
    y = torch.as_tensor(np.argmax(y, axis=1))
    features = torch.as_tensor(X)
    labels = y
    num_nodes = features.shape[0]

    dataset.graph = {'edge_index': None, 'edge_feat': None, 'node_feat': features, 'num_nodes': num_nodes}
    dataset.num_nodes = len(labels)
    dataset.label = torch.LongTensor(labels)
    edge_index = edge_index = build_graph(ref_adata, knears=knn_num)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long)
    dataset.edge_index = dataset.graph['edge_index'] = edge_index
    dataset.x = features
    # if ignore some class.  not fully match
    if ignore_first:
        dataset.label[dataset.label == 0] = -1
    dataset.splits = {
        'train': ref_adata.uns['train_idxs'],
        'valid': ref_adata.uns['val_idxs'],
        'test': ref_adata.uns['test_idxs']
    }

    ref_adata.var['gene_name'] = ref_adata.var.index
    return dataset, ref_adata


def load_cell_graph_fixed(ref_adata, ref_adata_name, runs):
    """
    given the dataset split data into: ID (train,val,test) and OOD.
    Since we do not use OOD train, we make it equal to OOD test
    """
    dataset, ref_adata = load_dataset_fixed(ref_adata, ref_adata_name, ood=True)
    dataset.y = dataset.label
    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset  # in distribution dataset
    number_class = dataset.y.max().item() + 1
    print('number of classes', number_class)
    # class_t = number_class//2-1  # ood training classes
    dataset_ind_list, dataset_ood_tr_list, dataset_ood_te_list = [], [], []
    for run in range(runs):
        train_idx, val_idx, test_idx = ref_adata.uns['train_idxs'][str(run)], ref_adata.uns['val_idxs'][str(
            run)], ref_adata.uns['test_idxs'][str(run)]
        ood_idx = ref_adata.uns["ood_idxs"][str(run)]
        id_idx = ref_adata.uns["id_idxs"][str(run)]

        dataset_ind.node_idx = id_idx
        dataset_ind.splits = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        dataset_ood_tr = Data(x=dataset.graph['node_feat'], edge_index=dataset.graph['edge_index'], y=dataset.y)
        dataset_ood_te = Data(x=dataset.graph['node_feat'], edge_index=dataset.graph['edge_index'], y=dataset.y)

        dataset_ood_tr.node_idx = dataset_ood_te.node_idx = ood_idx
        dataset_ind_list.append(dataset_ind)
        dataset_ood_tr_list.append(dataset_ood_tr)
        dataset_ood_te_list.append(dataset_ood_te)
    return dataset_ind_list, dataset_ood_tr_list, dataset_ood_te_list, ref_adata


def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)


class ZINBLoss(nn.Module):
    """ZINB loss class."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mean, disp, pi, scale_factor, ridge_lambda=0.0):
        """Forward propagation.

        Parameters
        ----------
        x :
            input features.
        mean :
            data mean.
        disp :
            data dispersion.
        pi :
            data dropout probability.
        scale_factor : list
            scale factor of mean.
        ridge_lambda : float optional
            ridge parameter.

        Returns
        -------
        result : float
            ZINB loss.

        """
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=.5):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class HetConv(nn.Module):
    """Neighborhood aggregation step."""

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class ZINBDecoder(nn.Module):
    """
    Parameters
    ----------
    input_dim : int
        dimension of input feature.
    n_z : int
        dimension of latent embedding.
    n_dec_1 : int optional
        number of nodes of decoder layer 1.
    n_dec_2 : int optional
        number of nodes of decoder layer 2.
    n_dec_3 : int optional
        number of nodes of decoder layer 3.

    """

    def __init__(self, input_dim, n_z, n_dec_1=128, n_dec_2=256, n_dec_3=512):
        super().__init__()
        self.n_dec_3 = n_dec_3
        self.input_dim = input_dim
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_3)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.dec_mean = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), nn.Sigmoid())

    def forward(self, z):
        """Forward propagation.

        Parameters
        ----------
        z :
            embedding.

        Returns
        -------
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB.

        """
        dec_h1 = F.relu(self.dec_1(z))
        dec_h3 = F.relu(self.dec_2(dec_h1))
        # dec_h3 = F.relu(self.dec_3(dec_h2))

        _mean = self.dec_mean(dec_h3)
        _disp = self.dec_disp(dec_h3)
        _pi = self.dec_pi(dec_h3)
        return _mean, _disp, _pi


class MeanAct(nn.Module):
    """Mean activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Dispersion activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class HeteroNet(nn.Module):
    """Our implementation with ZINB."""

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes, num_layers=2, dropout=0.5,
                 save_mem=False, num_mlp_layers=1, use_bn=True, conv_dropout=True, dec_dim=[], device="cpu"):
        super().__init__()

        self.feature_embed = MLP(in_channels, hidden_channels, hidden_channels, num_layers=num_mlp_layers,
                                 dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(HetConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(HetConv())
            if l != num_layers - 2:
                self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels * (2**(num_layers + 1) - 1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

        self.ZINB = ZINBDecoder(in_channels, last_dim, n_dec_1=dec_dim[0], n_dec_2=dec_dim[1], n_dec_3=dec_dim[2])
        self.device = device

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """Cache normalized adjacency and normalized strict two-hop adjacency, neither
        has self loops."""
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, x, edge_index, decoder=False):

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(x)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        if decoder:
            _mean, _disp, _pi = self.ZINB(x)
            h = self.final_project(x)
            return h, _mean, _disp, _pi
        x = self.final_project(x)
        return x


class scHeteroNet(nn.Module, BaseClassificationMethod):

    def __init__(self, d, c, edge_index, num_nodes, hidden_channels, num_layers, dropout, use_bn, device, min_loss):
        super().__init__()
        self.device = device
        self.encoder = HeteroNet(d, hidden_channels, c, edge_index=edge_index, num_nodes=num_nodes,
                                 num_layers=num_layers, dropout=dropout, use_bn=use_bn, dec_dim=[32, 64, 128])
        self.to(device)
        self.min_loss = min_loss

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def preprocessing_pipeline():
        pass

    def predict(self, x):
        return super().predict(x)

    def forward(self, dataset):
        """Return predicted logits."""
        x, edge_index = dataset.x.to(self.device), dataset.edge_index.to(self.device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        """Energy belief propagation, return the energy after propagation."""
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))  # normolized adjacency matrix
        for _ in range(prop_layers):  # iterative propagation
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def two_hop_propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))  # normalized adjacency matrix

        # Compute the two-hop adjacency matrix
        adj_2hop = adj @ adj

        for _ in range(prop_layers):  # iterative propagation
            e = e * alpha + matmul(adj_2hop, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, T, use_prop, use_2hop, oodprop, oodalpha):
        """Return negative energy, a vector for all input nodes."""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if dataset in ('proteins', 'ppi'):  # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = T * torch.logsumexp(logits / T, dim=-1).sum(dim=1)
        else:  # for single-label multi-class classification
            neg_energy = T * torch.logsumexp(logits / T, dim=-1)
        if use_prop:  # use energy belief propagation
            if use_2hop:
                neg_energy = self.two_hop_propagation(neg_energy, edge_index, oodprop, oodalpha)
            else:
                neg_energy = self.propagation(neg_energy, edge_index, oodprop, oodalpha)
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, use_zinb):
        """Return loss for training."""
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)
        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx
        logits_in, _mean, _disp, _pi = (
            i[train_in_idx]
            for i in self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device), decoder=use_zinb))
        logits_out = self.encoder(x_out, edge_index_out)
        # compute supervised training loss
        pred_in = F.log_softmax(logits_in, dim=1)
        sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))
        loss = sup_loss

        return loss, _mean, _disp, _pi, train_in_idx, logits_in

    def fit(self, dataset_ind, dataset_ood_tr, use_zinb, data, zinb_weight, cl_weight, mask_ratio, criterion,
            optimizer):
        adata = data.data
        self.train()
        optimizer.zero_grad()
        loss, _mean, _disp, _pi, train_idx, logit_in = self.loss_compute(dataset_ind, dataset_ood_tr, criterion,
                                                                         self.device, use_zinb)
        if use_zinb:
            zinb_loss = ZINBLoss().to(self.device)
            x_raw = adata.raw.X
            if scipy.sparse.issparse(x_raw):
                x_raw = x_raw.toarray()
            x_raw = torch.Tensor(x_raw)[train_idx].to(self.device)
            zinb_loss = zinb_loss(x_raw, _mean, _disp, _pi,
                                  torch.tensor(adata.obs.size_factors)[train_idx].to(self.device))
            loss += zinb_weight * zinb_loss
        if cl_weight != 0:
            X = dataset_ind.x.to(self.device)
            mask1 = (torch.rand_like(X) > mask_ratio).float()
            X_view1 = X * mask1
            z1 = self.encoder(X_view1, dataset_ind.edge_index.to(self.device))
            cl_loss = contrastive_loss(logit_in, z1)
            loss = loss + cl_weight * cl_loss
        loss.backward()
        optimizer.step()
        return loss

    # predict_proba and predict are used for the single run
    def predict_proba(self, dataset_ind):
        self.eval()
        with torch.no_grad():
            logits = self(dataset_ind)
            probabilities = F.softmax(logits, dim=1).cpu()
        return probabilities

    def predict(self, dataset_ind):
        probabilities = self.predict_proba(dataset_ind)
        predicted_labels = torch.argmax(probabilities, dim=1)
        return predicted_labels

    def evaluate(self, dataset_ind, dataset_ood_te, criterion, eval_func, display_step, run, results, epoch, loss,
                 dataset, T, use_prop, use_2hop, oodprop, oodalpha):
        result, test_ind_score, test_ood_score, representations = self.evaluate_detect(
            dataset_ind, dataset_ood_te, criterion, eval_func, self.device, return_score=True, dataset=dataset, T=T,
            use_prop=use_prop, use_2hop=use_2hop, oodprop=oodprop, oodalpha=oodalpha)
        results[run].append(result)
        if result[-1] < self.min_loss:
            self.min_loss = result[-1]
            selected_run = run
            selected_representation = representations.copy()
            selected_full_preds = selected_representation.argmax(axis=-1)
            selected_test_ind_score = test_ind_score.numpy().copy()
            selected_test_ood_score = test_ood_score.numpy().copy()
        if epoch % display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'AUROC: {100 * result[0]:.2f}%, '
                  f'AUPR: {100 * result[1]:.2f}%, '
                  f'FPR95: {100 * result[2]:.2f}%, '
                  f'Test Score: {100 * result[-2]:.2f}%')
        return result

    def score(self, x, y, *, score_func: Optional[Union[str, Mapping[Any, float]]] = None,
              return_pred: bool = False) -> Union[float, Tuple[float, Any]]:
        y_pred = self.predict(x)
        func = partial(eval_acc, acc=resolve_score_func(score_func or self._DEFAULT_METRIC))
        score = func(y, y_pred)
        return (score, y_pred) if return_pred else score

    # @torch.no_grad()  # this seems quite important, the orignal impl donot set this.
    def evaluate_detect(self, dataset_ind, dataset_ood, criterion, eval_func, device, return_score, dataset, T,
                        use_prop, use_2hop, oodprop, oodalpha, score_func: Optional[Union[str, Mapping[Any,
                                                                                                       float]]] = None):
        self.eval()

        with torch.no_grad():
            test_ind_score = self.detect(dataset_ind, dataset_ind.splits['test'], device, T, use_prop, use_2hop,
                                         oodprop, oodalpha).cpu()
        if isinstance(dataset_ood, list):
            result = []
            for d in dataset_ood:
                with torch.no_grad():
                    test_ood_score = self.detect(d, d.node_idx, device, T, use_prop, use_2hop, oodprop, oodalpha).cpu()
                auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
                result += [auroc] + [aupr] + [fpr]
        else:
            with torch.no_grad():
                test_ood_score = self.detect(dataset_ood, dataset_ood.node_idx, device, T, use_prop, use_2hop, oodprop,
                                             oodalpha).cpu()
            # print(test_ind_score, test_ood_score)
            auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
            result = [auroc] + [aupr] + [fpr]

        with torch.no_grad():
            out = self(dataset_ind).cpu()
            test_idx = dataset_ind.splits['test']
            test_score = eval_func(dataset_ind.y[test_idx], out[test_idx],
                                   acc=resolve_score_func(score_func or self._DEFAULT_METRIC))

            valid_idx = dataset_ind.splits['valid']
            if dataset in ('proteins', 'ppi'):
                valid_loss = criterion(out[valid_idx], dataset_ind.y[valid_idx].to(torch.float))
            else:
                valid_out = F.log_softmax(out[valid_idx], dim=1)
                valid_loss = criterion(valid_out, dataset_ind.y[valid_idx].squeeze(1))

            result += [test_score] + [valid_loss]

            if return_score:
                return result, test_ind_score, test_ood_score, out.detach().cpu().numpy()
            else:
                return result


# def ood_test(run,dataset_ind_list, dataset_ood_tr_list, dataset_ood_te_list,device,hidden_channels,num_layers,dropout,use_bn,lr,weight_decay,epochs,use_zinb,zinb_weight,cl_weight,mask_ratio,criterion,eval_func,display_step,adata,le,results):
#         dataset_ind, dataset_ood_tr, dataset_ood_te = dataset_ind_list[run], dataset_ood_tr_list[run], dataset_ood_te_list[run]
#         if len(dataset_ind.y.shape) == 1:
#             dataset_ind.y = dataset_ind.y.unsqueeze(1)
#         if len(dataset_ood_tr.y.shape) == 1:
#             dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
#         if isinstance(dataset_ood_te, list):
#             for data in dataset_ood_te:
#                 if len(data.y.shape) == 1:
#                     data.y = data.y.unsqueeze(1)
#         else:
#             if len(dataset_ood_te.y.shape) == 1:
#                 dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

#         c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
#         d = dataset_ind.graph['node_feat'].shape[1]
#         model = scHeteroNet(d, c, dataset_ind.edge_index.to(device), dataset_ind.num_nodes,hidden_channels=hidden_channels,num_layers=num_layers,dropout=dropout,use_bn=use_bn).to(device)
#         criterion = nn.NLLLoss()
#         model.train()
#         model.reset_parameters()
#         model.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         for epoch in range(epochs):
#             loss=model.fit(dataset_ind, dataset_ood_tr, use_zinb, adata,zinb_weight, cl_weight, mask_ratio, criterion, optimizer)
#             model.evaluate(dataset_ind, dataset_ood_te, criterion, eval_func, display_step, le, run, results, epoch, loss)
#         print_statistics(results,run)

# def eval_acc(y_true, y_pred):
#     acc_list = []
#     y_true = y_true.detach().cpu().numpy()
#     if y_pred.shape == y_true.shape:
#         y_pred = y_pred.detach().cpu().numpy()
#     else:
#         y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

#     for i in range(y_true.shape[1]):
#         is_labeled = y_true[:, i] == y_true[:, i]
#         correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
#         acc_list.append(float(np.sum(correct))/len(correct))

#     return sum(acc_list)/len(acc_list)
import anndata as ad


def get_genename(raw_adata):
    if "gene_id" in raw_adata.var.keys():
        gene_name = raw_adata.var["gene_id"].values
    elif "symbol" in raw_adata.var.keys():
        gene_name = raw_adata.var["symbol"].values
    else:
        gene_name = raw_adata.var.index
    return gene_name


# def set_idx_split(data,train_idxs = [],
#     val_idxs = [],
#     test_idxs = []):
#     adata=data.data
#     raw_adata = adata
#     y = np.argmax(raw_adata.obsm["cell_type"], axis=1)
#     for obsm in raw_adata.obsm.keys():
#         if obsm in ["cell_type"]:
#             adata.obs[obsm + "_raw"] = np.argmax(raw_adata.obsm[obsm].values , axis=1)
#         print("copy", obsm)
#         adata.obsm[obsm] = raw_adata.obsm[obsm].values
#     adata.obs["cell"] = y
#     adata.var["gene_name"] = get_genename(raw_adata)

#     ood_idxs = []
#     id_idxs = []
#     ood_class = min(Counter(y).items(), key=lambda x: x[1])[0]
#     ood_idx = [i for i, value in enumerate(y) if value == ood_class]
#     id_idx = [i for i, value in enumerate(y) if value != ood_class]
#     ood_idxs.append(ood_idx)
#     id_idxs.append(id_idx)
#     adata.uns["train_idxs"] = {str(key): value for key, value in enumerate(train_idxs)}
#     adata.uns["val_idxs"] = {str(key): value for key, value in enumerate(val_idxs)}
#     adata.uns["test_idxs"] = {str(key): value for key, value in enumerate(test_idxs)}
#     adata.uns["ood_idxs"] = {str(key): value for key, value in enumerate(ood_idxs)}
#     adata.uns["id_idxs"] = {str(key): value for key, value in enumerate(id_idxs)}
#     adata.obs['n_counts']= adata.X.sum(axis=1)
#     adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)


def set_split(data, seeds=[42, 66, 88, 2023, 2024]):
    adata = data.data
    raw_adata = adata
    y = np.argmax(raw_adata.obsm["cell_type"], axis=1)
    for obsm in raw_adata.obsm.keys():
        if obsm in ["cell_type"]:
            adata.obs[obsm + "_raw"] = np.argmax(raw_adata.obsm[obsm].values, axis=1)
        print("copy", obsm)
        adata.obsm[obsm] = raw_adata.obsm[obsm].values
    adata.obs["cell"] = y
    adata.var["gene_name"] = get_genename(raw_adata)
    train_idxs = []
    val_idxs = []
    test_idxs = []
    ood_idxs = []
    id_idxs = []
    for seed in seeds:
        ood_class = min(Counter(y).items(), key=lambda x: x[1])[0]
        ood_idx = [i for i, value in enumerate(y) if value == ood_class]
        id_idx = [i for i, value in enumerate(y) if value != ood_class]
        full_indices = np.arange(adata.shape[0])
        train_idx, test_idx = train_test_split(full_indices, test_size=0.2, random_state=seed)
        train_val_indices = train_idx
        train_idx, val_idx = train_test_split(train_val_indices, test_size=0.25, random_state=seed)
        train_idx = [i for i in train_idx if i not in ood_idx]
        val_idx = [i for i in val_idx if i not in ood_idx]
        test_idx = [i for i in test_idx if i not in ood_idx]
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)
        test_idxs.append(test_idx)
        ood_idxs.append(ood_idx)
        id_idxs.append(id_idx)
    adata.uns["train_idxs"] = {str(key): value for key, value in enumerate(train_idxs)}
    adata.uns["val_idxs"] = {str(key): value for key, value in enumerate(val_idxs)}
    adata.uns["test_idxs"] = {str(key): value for key, value in enumerate(test_idxs)}
    adata.uns["ood_idxs"] = {str(key): value for key, value in enumerate(ood_idxs)}
    adata.uns["id_idxs"] = {str(key): value for key, value in enumerate(id_idxs)}
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)


# import scanpy as sc
# import numpy as np
# import anndata
# import random

# def normalize_adata(adata, size_factors=True, normalize_input=True, logtrans_input=True):
#     if size_factors or normalize_input or logtrans_input:
#         adata.raw = adata.copy()
#     else:
#         adata.raw = adata
#     if size_factors:
#         sc.pp.normalize_per_cell(adata, min_counts=0)
#         adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
#     else:
#         adata.obs['size_factors'] = 1.0

#     if logtrans_input:
#         sc.pp.log1p(adata)

#     if normalize_input:
#         sc.pp.scale(adata)

#     return adata

# def filter_cellType(adata):
#     adata_copied = adata.copy()
#     cellType_Number = adata_copied.obs.cell.value_counts()
#     celltype_to_remove = cellType_Number[cellType_Number <= 10].index
#     adata_copied = adata_copied[~adata_copied.obs.cell.isin(celltype_to_remove), :]

#     return adata_copied

# def filter_data(X, highly_genes=4000):
#     X = np.ceil(X).astype(int)
#     adata = sc.AnnData(X, dtype=np.float32)
#     sc.pp.filter_genes(adata, min_counts=3)
#     sc.pp.filter_cells(adata, min_counts=1)
#     sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, flavor='cell_ranger', min_disp=0.5,
#                                 n_top_genes=highly_genes, subset=True)
#     genes_idx = np.array(adata.var_names.tolist()).astype(int)
#     cells_idx = np.array(adata.obs_names.tolist()).astype(int)

#     return genes_idx, cells_idx

# def gen_split(dataset, normalize_input=False, logtrans_input=True):     #
#     raw_adata = anndata.read_h5ad('./data/'+dataset+'.h5ad')
#     raw_adata.obs['cell'] = raw_adata.obs['cell_type']
#     # delete cell_type column
#     if 'cell_type' in raw_adata.obs.keys():
#         del raw_adata.obs['cell_type']
#     raw_adata = raw_adata[raw_adata.obs['assay'] == '10x 3\' v2']
#     print("filtering cells whose cell type number is less than 10")
#     raw_adata = filter_cellType(raw_adata)
#     # fileter and normalize (bio processing)
#     X = raw_adata.X.toarray()
#     y = raw_adata.obs["cell"]
#     genes_idx, cells_idx = filter_data(X)
#     X = X[cells_idx][:, genes_idx]
#     y = y[cells_idx]
#     adata = sc.AnnData(X, dtype=np.float32)
#     # sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
#     # sc.pp.log1p(adata)
#     adata = normalize_adata(adata, size_factors=True, normalize_input=normalize_input, logtrans_input=logtrans_input)
