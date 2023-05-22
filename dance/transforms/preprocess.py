import math
import os
import random
import time
import warnings

import anndata
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.utils.extmath
import torch
from dgl.sampling import pack_traces, random_walk
from scipy.stats import expon
from sklearn.model_selection import train_test_split

from dance.utils.deprecate import deprecated


def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prefilter_cells(adata, min_counts=None, max_counts=None, min_genes=200, max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[0], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, min_genes=min_genes)[0]) if min_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, max_genes=max_genes)[0]) if max_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw = sc.pp.log1p(adata, copy=True)  # check the rowname
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:", adata.raw.var_names.is_unique)


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


def prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)


@deprecated
def normalize(adata, counts_per_cell_after=1e4, log_transformed=False):
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=counts_per_cell_after)


@deprecated
def normalize_total(adata, target_sum=1e4):
    sc.pp.normalize_total(adata, target_sum=target_sum)


@deprecated
def log1p(adata):
    sc.pp.log1p(adata)


def calculate_log_library_size(Dataset):
    # Dataset is raw read counts, and should be cells * features

    Nsamples = np.shape(Dataset)[0]
    library_sum = np.log(np.sum(Dataset, axis=1))

    lib_mean = np.full((Nsamples, 1), np.mean(library_sum))
    lib_var = np.full((Nsamples, 1), np.var(library_sum))

    return lib_mean, lib_var


class lsiTransformer():

    def __init__(
        self,
        n_components: int = 20,
        drop_first=True,
    ):

        self.drop_first = drop_first
        self.n_components = n_components + drop_first
        self.tfidfTransformer = tfidfTransformer()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(n_components=self.n_components, random_state=777)
        self.fitted = None

    def fit(self, adata: anndata.AnnData):
        X = self.tfidfTransformer.fit_transform(adata.layers['counts'])
        X_norm = self.normalizer.fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        self.pcaTransformer.fit(X_norm)
        self.fitted = True

    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        X = self.tfidfTransformer.transform(adata.layers['counts'])
        X_norm = self.normalizer.transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = self.pcaTransformer.transform(X_norm)
        #         X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        #         X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
        lsi_df = pd.DataFrame(X_lsi, index=adata.obs_names).iloc[:, int(self.drop_first):]
        return lsi_df

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)


class tfidfTransformer():

    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / X.sum(axis=0)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            return tf.multiply(self.idf)
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


####################################
# Copied from GraphSaint (https://github.com/lt610/GraphSaint)
####################################


# The base class of sampler
# (TODO): online sampling
class SAINTSampler:

    def __init__(self, dn, g, train_nid, node_budget, num_repeat=50):
        """:param dn: name of dataset :param g: full graph :param train_nid: ids of
        training nodes :param node_budget: expected number of sampled nodes :param
        num_repeat: number of times of repeating sampling one node."""
        self.g = g
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_repeat = dn, num_repeat
        self.node_counter = torch.zeros((self.train_g.num_nodes(), ))
        self.edge_counter = torch.zeros((self.train_g.num_edges(), ))
        self.prob = None

        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs('./subgraphs/', exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0

            t = time.perf_counter()
            while sampled_nodes <= self.train_g.num_nodes() * num_repeat:
                subgraph = self.__sample__()
                self.subgraphs.append(subgraph)
                sampled_nodes += subgraph.shape[0]
                self.N += 1
            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            np.save(graph_fn, self.subgraphs)

            t = time.perf_counter()
            self.__counter__()
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f'Normalization time: [{time.perf_counter() - t:.2f}s]')
            np.save(norm_fn, (aggr_norm, loss_norm))

        self.train_g.ndata['l_n'] = torch.Tensor(loss_norm)
        self.train_g.edata['w'] = torch.Tensor(aggr_norm)
        self.__compute_degree_norm()

        self.num_batch = math.ceil(self.train_g.num_nodes() / node_budget)
        random.shuffle(self.subgraphs)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))
        print("The size of subgraphs is about: ", len(self.subgraphs[-1]))

    def __clear__(self):
        self.prob = None
        self.node_counter = None
        self.edge_counter = None
        self.g = None

    def __counter__(self):

        for sampled_nodes in self.subgraphs:
            sampled_nodes = torch.from_numpy(sampled_nodes)
            self.node_counter[sampled_nodes] += 1

            subg = self.train_g.subgraph(sampled_nodes)
            sampled_edges = subg.edata[dgl.EID]
            self.edge_counter[sampled_edges] += 1

    def __generate_fn__(self):
        raise NotImplementedError

    def __compute_norm__(self):
        self.node_counter[self.node_counter == 0] = 1
        self.edge_counter[self.edge_counter == 0] = 1

        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()

        self.train_g.ndata['n_c'] = self.node_counter
        self.train_g.edata['e_c'] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'a_n'))
        aggr_norm = self.train_g.edata.pop('a_n')

        self.train_g.ndata.pop('n_c')
        self.train_g.edata.pop('e_c')

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):

        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __sample__(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batch:
            result = self.train_g.subgraph(self.subgraphs[self.n])
            self.n += 1
            return result
        else:
            random.shuffle(self.subgraphs)
            raise StopIteration()


class SAINTRandomWalkSampler(SAINTSampler):

    def __init__(self, num_roots, length, dn, g, train_nid, num_repeat=50):
        self.num_roots, self.length = num_roots, length
        super().__init__(dn, g, train_nid, num_roots * length, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots, self.length,
                                                                        self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots, self.length,
                                                                            self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = torch.randint(0, self.train_g.num_nodes(), (self.num_roots, ))
        traces, types = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()


def l2norm(mat):
    stat = np.sqrt(np.sum(mat**2, axis=1))
    cols = mat.columns
    mat[cols] = mat[cols].div(stat, axis=0)
    mat[np.isinf(mat)] = 0
    return mat


def ccaEmbed(data1, data2, num_cc=30):
    random.seed(123)
    object1 = sklearn.preprocessing.scale(data1)
    object2 = sklearn.preprocessing.scale(data2)
    mat3 = np.matmul(np.matrix(object1).transpose(), np.matrix(object2))
    a = SVD(mat=mat3, num_cc=int(num_cc))
    embeds_data = np.concatenate((a[0], a[1]))
    ind = np.where([embeds_data[:, col][0] < 0 for col in range(embeds_data.shape[1])])[0]
    embeds_data[:, ind] = embeds_data[:, ind] * (-1)

    embeds_data = pd.DataFrame(embeds_data)
    embeds_data.index = np.concatenate((np.array(data1.columns), np.array(data2.columns)))
    embeds_data.columns = ['D_' + str(i) for i in range(num_cc)]
    d = a[2]

    cell_embeddings = np.matrix(embeds_data)
    combined_data = data1.merge(data2, left_index=True, right_index=True, how='inner')
    new_data1 = combined_data.dropna()
    loadings = pd.DataFrame(np.matmul(np.matrix(new_data1), cell_embeddings))
    loadings.index = new_data1.index
    return [embeds_data, d], loadings


def sortGenes(Loadings, dim, numG):
    data = Loadings.iloc[:, dim]
    num = np.round(numG / 2).astype('int')
    data1 = data.sort_values(ascending=False)
    data2 = data.sort_values(ascending=True)
    posG = np.array(data1.index[0:num])
    negG = np.array(data2.index[0:num])
    topG = np.concatenate((posG, negG))
    return topG


def selectTopGenes(Loadings, dims, DimGenes, maxGenes):
    maxG = max(len(dims) * 2, maxGenes)
    gens = [None] * DimGenes
    idx = -1
    for i in range(1, DimGenes + 1):
        idx = idx + 1
        selg = []
        for j in dims:
            selg.extend(set(sortGenes(Loadings, dim=j, numG=i)))
        gens[idx] = set(selg)
    lens = np.array([len(i) for i in gens])
    lens = lens[lens < maxG]
    maxPer = np.where(lens == np.max(lens))[0][0] + 1
    selg = []
    for j in dims:
        selg.extend(set(sortGenes(Loadings, dim=j, numG=maxPer)))
    selgene = np.array(list(set(selg)), dtype=object)
    return (selgene)


def filter_data(data, highly_genes=500):
    adata = data.data.copy()
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, flavor='cell_ranger', min_disp=0.5,
                                n_top_genes=highly_genes, subset=True)
    data._data = data.data[adata.obs_names, adata.var_names]


def generate_random_pair(y, label_cell_indx, num, error_rate=0):
    """Generate random pairwise constraints."""
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = np.array(y)

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
            if ind1 == l1 and ind2 == l2:
                return True
        return False

    error_num = 0
    num0 = num
    while num > 0:
        tmp1 = random.choice(label_cell_indx)
        tmp2 = random.choice(label_cell_indx)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if y[tmp1] == y[tmp2]:
            if error_num >= error_rate * num0:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
            else:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
                error_num += 1
        else:
            if error_num >= error_rate * num0:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
            else:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
                error_num += 1
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num


def geneSelection(data, threshold=0, atleast=10, yoffset=.02, xoffset=5, decay=1.5, n=None, verbose=1):
    if sp.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (1 - zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:, detected] > threshold
        logs = np.zeros_like(data[:, detected]) * np.nan
        logs[mask] = np.log2(data[:, detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        if verbose > 0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
    return selected


def load_graph(path, data):
    """Load graph for scDSC."""
    n = data.shape[0]
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = scipy.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)

    # Construct a symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + np.eye(adj.shape[0])
    adj = scipy.sparse.coo_matrix(row_normalize(adj), dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize_adata(data, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(data.data, min_counts=1)
        sc.pp.filter_cells(data.data, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        data.data.raw = data.data.copy()
    else:
        data.data.raw = data.data

    if size_factors:
        sc.pp.normalize_per_cell(data.data)
        data.data.obs['size_factors'] = data.data.obs.n_counts / np.median(data.data.obs.n_counts)
    else:
        data.data.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(data.data)

    if normalize_input:
        sc.pp.scale(data.data)


def row_normalize(mx):
    # Row-normalize sparse matrix
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert a scipy sparse matrix to a torch sparse tensor.
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def SVD(mat, num_cc):
    U, s, V = np.linalg.svd(mat)
    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d


##############################
# Imputation Data Masking
##############################


class MaskedArray:

    def __init__(self, data=None, mask=None, distr="exp", dropout=0.01, seed=1):
        self.data = np.array(data)
        self._binMask = np.array(mask)
        self.shape = data.shape
        self.distr = distr
        self.dropout = dropout
        self.seed = seed

    @property
    def binMask(self):
        return self._binMask

    @binMask.setter
    def binMask(self, value):
        self._binMask = value.astype(bool)

    def getMaskedMatrix(self):
        maskedMatrix = self.data.copy()
        maskedMatrix[~self.binMask] = 0
        return maskedMatrix

    def getMasked(self, rows=True):
        """Generator for row or column mask."""
        compt = 0
        if rows:
            while compt < self.shape[0]:
                yield [self.data[compt, idx] for idx in range(self.shape[1]) if not self.binMask[compt, idx]]
                compt += 1
        else:
            while compt < self.shape[1]:
                yield [self.data[idx, compt] for idx in range(self.shape[0]) if not self.binMask[idx, compt]]
                compt += 1

    def getMasked_flat(self):
        return self.data[~self.binMask]

    def copy(self):
        args = {"data": self.data.copy(), "mask": self.binMask.copy()}
        return MaskedArray(**args)

    def get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)

    def get_Nmasked(self, idx):
        cells_g = self.data[:, idx]
        dp_i = (1 + (cells_g == 0).sum() * 1.) / self.shape[0]
        dp_f = np.exp(-2 * np.log10(cells_g.mean())**2)
        return 1 + int((cells_g == 0).sum() * dp_f / dp_i)

    def generate(self):
        np.random.seed(self.seed)
        self.binMask = np.ones(self.shape).astype(bool)

        for c in range(self.shape[0]):
            cells_c = self.data[c, :]
            # Retrieve indices of positive values
            ind_pos = np.arange(self.shape[1])[cells_c > 0]
            cells_c_pos = cells_c[ind_pos]
            # Get masking probability of each value

            if cells_c_pos.size > 5:
                probs = self.get_probs(cells_c_pos)
                n_masked = 1 + int(self.dropout * len(cells_c_pos))
                if n_masked >= cells_c_pos.size:
                    print("Warning: too many cells masked for gene {} ({}/{})".format(c, n_masked, cells_c_pos.size))
                    n_masked = 1 + int(0.5 * cells_c_pos.size)

                masked_idx = np.random.choice(cells_c_pos.size, n_masked, p=probs / probs.sum(), replace=False)
                self.binMask[c, ind_pos[sorted(masked_idx)]] = False


##############################
# Imputation Data Loading
##############################


def load_imputation_data_internal(params, model_params, model):
    random_seed = params.random_seed
    if params.train_dataset == 'mouse_embryo' or params.train_dataset == 'mouse_embryo_data':
        if params.train_dataset[-5:] != '_data':
            train_dataset = params.train_dataset + '_data'
        else:
            train_dataset = params.train_dataset
        for i in range(len(params.dataset_to_file[train_dataset])):
            fname = params.dataset_to_file[train_dataset][i]
            data_path = f'{params.data_dir}/train/{train_dataset}/{fname}'
            if i == 0:
                counts = pd.read_csv(data_path, header=None, index_col=0)
                time = pd.Series(np.zeros(counts.shape[1]))
                # counts = pd.concat([counts, time], axis= 1)
            else:
                x = pd.read_csv(data_path, header=None, index_col=0)
                time = pd.concat([time, pd.Series(np.zeros(x.shape[1])) + i])
                # x = pd.concat([x, time], axis=1)
                counts = pd.concat([counts, x], axis=1)
        time = pd.DataFrame(time)
        time.columns = ['time']
        counts = counts[counts.sum(axis=1) != 0]
        counts = counts.T
        counts.index = [i for i in range(counts.shape[0])]
        adata = sc.AnnData(counts.values)
        adata.var_names = counts.columns.tolist()
        adata.obs['time'] = time.to_numpy()
    else:
        if params.train_dataset[-5:] != '_data':
            train_dataset = params.train_dataset + '_data'
        else:
            train_dataset = params.train_dataset
        data_path = f'{params.data_dir}/train/{train_dataset}/{params.dataset_to_file[train_dataset]}'
        # graph_path = 'pretrained/graphs'
        if not os.path.exists(data_path):
            raise NotImplementedError

        if params.filetype == 'csv' or params.dataset_to_file[train_dataset][-3:] == 'csv':
            counts = pd.read_csv(data_path, index_col=0, header=None)
            counts = counts[counts.sum(axis=1) != 0]
            counts = counts.T
            adata = sc.AnnData(counts.values)
            # adata.obs_names = ["%d"%i for i in range(adata.shape[0])]
            adata.obs_names = counts.index.tolist()
            adata.var_names = counts.columns.tolist()
        if params.filetype == 'gz' or params.dataset_to_file[train_dataset][-2:] == 'gz':
            counts = pd.read_csv(data_path, index_col=0, compression='gzip', header=0)
            counts = counts[counts.sum(axis=1) != 0]
            counts = counts.T
            adata = sc.AnnData(counts.values)
            # adata.obs_names = ["%d" % i for i in range(adata.shape[0])]
            adata.obs_names = counts.index.tolist()
            adata.var_names = counts.columns.tolist()
        elif params.filetype == 'h5' or params.dataset_to_file[train_dataset][-2:] == 'h5':
            adata = sc.read_10x_h5(data_path)
            adata.var_names_make_unique()
            counts = pd.DataFrame(adata.X.toarray())
            counts.columns = adata.var_names
            counts.index = adata.obs_names

    if model == "DeepImpute":

        def inspect_data(data):
            # Check if there area any duplicated cell/gene labels

            if sum(data.index.duplicated()):
                print("ERROR: duplicated cell labels. Please provide unique cell labels.")
                exit(1)

            if sum(data.columns.duplicated()):
                print("ERROR: duplicated gene labels. Please provide unique gene labels.")
                exit(1)

            max_value = np.max(data.values)
            if max_value < 10:
                print(
                    "ERROR: max value = {}. Is your data log-transformed? Please provide raw counts".format(max_value))
                exit(1)

            print("Input dataset is {} cells (rows) and {} genes (columns)".format(*data.shape))
            print("First 3 rows and columns:")
            print(data.iloc[:3, :3])

        inspect_data(counts)

        ### set masked data ###
        true_counts = counts
        maskedData = MaskedArray(data=true_counts)
        maskedData.generate()
        counts = pd.DataFrame(maskedData.getMaskedMatrix(), index=true_counts.index, columns=true_counts.columns)

        if model_params.cell_subset != 1:
            if model_params.cell_subset < 1:
                counts = counts.sample(frac=model_params.cell_subset)
            else:
                counts = counts.sample(model_params.cell_subset)

        gene_metric = (counts.var() / (1 + counts.mean())).sort_values(ascending=False)
        gene_metric = gene_metric[gene_metric > 0]

        ### Determine which genes will be imputed ###
        if model_params.genes_to_impute is None:
            if not str(model_params.NN_lim).isdigit():
                NN_lim = (gene_metric > model_params.minVMR).sum()
            else:
                NN_lim = model_params.NN_lim

            n_subsets = int(np.ceil(NN_lim / model_params.sub_outputdim))
            genes_to_impute = gene_metric.index[:n_subsets * model_params.sub_outputdim]

            rest = model_params.sub_outputdim - (len(genes_to_impute) % model_params.sub_outputdim)

            if rest > 0:
                fill_genes = np.random.choice(gene_metric.index, rest)
                genes_to_impute = np.concatenate([genes_to_impute, fill_genes])
            n_genes = len(genes_to_impute)
            print("{} genes selected for imputation".format(len(genes_to_impute)))
        else:
            # Make the number of genes to impute a multiple of the network output dim
            n_genes = len(model_params.genes_to_impute)
            if n_genes % model_params.sub_outputdim != 0:
                print("The number of input genes is not a multiple of {}. Filling with other genes.".format(n_genes))
                fill_genes = gene_metric.index[:model_params.sub_outputdim - n_genes]

                if len(fill_genes) < model_params.sub_outputdim - n_genes:
                    # Not enough genes in gene_metric. Sample with replacement
                    rest = model_params.sub_outputdim - n_genes - len(fill_genes)
                    fill_genes = np.concatenate([fill_genes, np.random.choice(gene_metric.index, rest, replace=True)])

                genes_to_impute = np.concatenate([model_params.genes_to_impute, fill_genes])
            else:
                genes_to_impute = model_params.genes_to_impute

        ### Get Covariance ###
        VMR = counts.std() / counts.mean()
        VMR[np.isinf(VMR)] = 0

        if model_params.n_pred is None:
            potential_pred = counts.columns[VMR > 0]
        else:
            print("Using {} predictors".format(model_params.n_pred))
            potential_pred = VMR.sort_values(ascending=False).index[:model_params.n_pred]

        covariance_matrix = pd.DataFrame(np.abs(np.corrcoef(counts.T.loc[potential_pred])), index=potential_pred,
                                         columns=potential_pred).fillna(0)
        ### get targets ###
        temp_counts = counts.reindex(columns=genes_to_impute)

        n_subsets = int(temp_counts.shape[1] / model_params.sub_outputdim)

        if model_params.mode == 'progressive':
            targets = temp_counts.columns.values.reshape([n_subsets, model_params.sub_outputdim])
        else:
            targets = np.random.choice(temp_counts.columns, [n_subsets, model_params.sub_outputdim], replace=False)
        ### get predictors ###
        predictors = []

        for i, targs in enumerate(targets):

            genes_not_in_target = np.setdiff1d(covariance_matrix.columns, targs)

            if genes_not_in_target.size == 0:
                warnings.warn(
                    'Warning: number of target genes lower than output dim. Consider lowering down the sub_outputdim parameter',
                    UserWarning)
                genes_not_in_target = covariance_matrix.columns

            subMatrix = (covariance_matrix.loc[targs, genes_not_in_target])
            sorted_idx = np.argsort(-subMatrix.values, axis=1)
            preds = subMatrix.columns[sorted_idx[:, :model_params.ntop].flatten()]

            predictors.append(preds.unique())

            print("Net {}: {} predictors, {} targets".format(i, len(np.unique(preds)), len(targs)))

        print("Normalization")
        norm_data = np.log1p(counts).astype(np.float32)  # normalizer.transform(raw)

        test_cells = np.random.choice(norm_data.index, int(0.05 * norm_data.shape[0]), replace=False)
        train_cells = np.setdiff1d(norm_data.index, test_cells)

        X_train = [norm_data.loc[train_cells, inputgenes].values for inputgenes in predictors]
        Y_train = [norm_data.loc[train_cells, targetgenes].values for targetgenes in targets]

        X_test = [norm_data.loc[test_cells, inputgenes].values for inputgenes in predictors]
        Y_test = [norm_data.loc[test_cells, targetgenes].values for targetgenes in targets]

        data_dict = {}
        data_dict['num_cells'] = norm_data.shape[0]
        data_dict['num_genes'] = n_genes
        data_dict['train_data'] = [X_train, Y_train]
        data_dict['test_data'] = [X_test, Y_test]
        data_dict['true_counts'] = true_counts
        data_dict['total_counts'] = counts
        data_dict['genes_to_impute'] = genes_to_impute
        data_dict['adata'] = adata
        data_dict['predictors'] = predictors
        data_dict['targets'] = targets

    return data_dict
