# Copyright 2022 DSE lab.  All rights reserved.
import collections
import os
import pprint
import random
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Union

import anndata
import cv2
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
import statsmodels.api as sm
import statsmodels.stats.multitest as smt
import torch
import torch.nn.functional as F
from anndata import AnnData
from dgl.sampling import pack_traces, random_walk
from scipy import stats
from scipy.sparse import csr_matrix, load_npz, save_npz, spmatrix, vstack
from scipy.stats import expon
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols


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
    adata.raw = sc.pp.log1p(adata, copy=True)  #check the rowname
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


def normalize(adata, counts_per_cell_after=1e4, log_transformed=False):
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=counts_per_cell_after)


def normalize_total(adata, target_sum=1e4):
    sc.pp.normalize_total(adata, target_sum=target_sum)


def log1p(adata):
    sc.pp.log1p(adata)


def calculate_log_library_size(Dataset):
    ### Dataset is raw read counts, and should be cells * features

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
        """
        :param dn: name of dataset
        :param g: full graph
        :param train_nid: ids of training nodes
        :param node_budget: expected number of sampled nodes
        :param num_repeat: number of times of repeating sampling one node
        """
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


#####################################
# Cell Type Annotation for ScDeepSort
#####################################


def get_map_dict(map_path: Path, tissue):
    map_df = pd.read_excel(os.path.join(map_path, 'map.xlsx'))

    # {num: {test_cell1: {train_cell1, train_cell2}, {test_cell2:....}}, num_2:{}...}
    map_dic = dict()
    for idx, row in enumerate(map_df.itertuples()):
        if getattr(row, 'Tissue') == tissue:
            num = getattr(row, 'num')
            test_celltype = getattr(row, 'Celltype')
            train_celltype = getattr(row, '_5')
            if map_dic.get(getattr(row, 'num')) is None:
                map_dic[num] = dict()
                map_dic[num][test_celltype] = set()
            elif map_dic[num].get(test_celltype) is None:
                map_dic[num][test_celltype] = set()
            map_dic[num][test_celltype].add(train_celltype)
    return map_dic


def normalize_weight(graph: dgl.DGLGraph):
    # normalize weight & add self-loop

    in_degrees = graph.in_degrees()
    print(in_degrees)
    print(graph.number_of_nodes())
    for i in range(graph.number_of_nodes()):
        src, dst, in_edge_id = graph.in_edges(i, form='all')
        if src.shape[0] == 0:
            continue
        edge_w = graph.edata['weight'][in_edge_id]
        graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(edge_w)


def get_id_to_gene(gene_statistics_path):
    id2gene = []
    with open(gene_statistics_path, encoding='utf-8') as f:
        for line in f:
            id2gene.append(line.strip())
    return id2gene


def get_id_to_label(cell_statistics_path):
    id2label = []
    with open(cell_statistics_path, encoding='utf-8') as f:
        for line in f:
            id2label.append(line.strip())
    return id2label


def load_annotation_test_data(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    test = params.test_dataset
    tissue = params.tissue

    proj_path = Path(params.proj_path)
    species_data_path = proj_path / 'pretrained' / params.species
    statistics_path = species_data_path / 'statistics'

    if params.score:
        map_path = proj_path / 'map' / params.species
        map_dict = get_map_dict(map_path, tissue)

    if not statistics_path.exists():
        statistics_path.mkdir()

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
    cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels

    # generate gene statistics file
    id2gene = get_id_to_gene(gene_statistics_path)
    # generate cell label statistics file
    id2label = np.array(get_id_to_label(cell_statistics_path), dtype=str)

    test_num = 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    print(f"The build graph contains {num_genes} gene nodes with {num_labels} labels supported.")

    test_graph_dict = dict()  # test-graph dict
    if params.score:
        test_label_dict = dict()  # test label dict
    test_index_dict = dict()  # test feature indices in all features
    test_mask_dict = dict()
    test_nid_dict = dict()
    test_cell_origin_id_dict = dict()

    ids = torch.arange(num_genes, dtype=torch.int32).unsqueeze(-1)

    # ==================================================
    # add all genes as nodes

    for num in test:
        test_graph_dict[num] = dgl.DGLGraph()
        test_graph_dict[num].add_nodes(num_genes, {'id': ids})
    # ====================================================

    matrices = []

    support_data = proj_path / 'pretrained' / f'{params.species}' / 'graphs' / f'{params.species}_{tissue}_data.npz'
    support_num = 0
    info = load_npz(support_data)
    print(f"load {support_data.name}")
    row_idx, gene_idx = np.nonzero(info > 0)
    non_zeros = info.data
    cell_num = info.shape[0]
    support_num += cell_num
    matrices.append(info)
    ids = torch.tensor([-1] * cell_num, dtype=torch.int32).unsqueeze(-1)
    total_cell = support_num

    for n in test:  # training cell also in test graph
        cell_idx = row_idx + test_graph_dict[n].number_of_nodes()
        test_graph_dict[n].add_nodes(cell_num, {'id': ids})
        test_graph_dict[n].add_edges(cell_idx, gene_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)})
        test_graph_dict[n].add_edges(gene_idx, cell_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)})

    for num in test:
        data_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_data.{params.filetype}'
        if params.score:
            type_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_celltype.csv'
            # load celltype file then update labels accordingly
            cell2type = pd.read_csv(type_path, index_col=0)
            cell2type.columns = ['cell', 'type']
            cell2type['type'] = cell2type['type'].map(str.strip)
            test_label_dict[num] = list(map(map_dict[num].get, cell2type['type'].tolist()))

        # load data file then update graph
        if params.filetype == 'csv':
            df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        elif params.filetype == 'gz':
            df = pd.read_csv(data_path, compression='gzip', index_col=0)
        else:
            print(f'Not supported type for {data_path}. Please verify your data file')

        test_cell_origin_id_dict[num] = list(df.columns)
        df = df.transpose(copy=True)  # (cell, gene)

        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(
            f'{params.species}_{tissue}{num}_data.{params.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%'
        )
        tic = time.time()
        print(f'Begin to cumulate time of training/testing ...')
        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        cell_idx = row_idx + test_graph_dict[num].number_of_nodes()
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        # test_nodes_index_dict[num] = list(range(graph.number_of_nodes(), graph.number_of_nodes() + len(df)))
        ids = torch.tensor([-1] * len(df), dtype=torch.int32).unsqueeze(-1)
        test_index_dict[num] = list(
            range(num_genes + support_num + test_num, num_genes + support_num + test_num + len(df)))
        test_nid_dict[num] = list(
            range(test_graph_dict[num].number_of_nodes(), test_graph_dict[num].number_of_nodes() + len(df)))
        test_num += len(df)
        test_graph_dict[num].add_nodes(len(df), {'id': ids})
        # for the test cells, only gene-cell edges are in the test graph
        test_graph_dict[num].add_edges(gene_idx, cell_idx,
                                       {'weight': torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        total_cell += num

    support_index = list(range(num_genes + support_num))
    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:support_num].T)
    gene_feat = gene_pca.transform(sparse_feat[:support_num].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
    for num in test:
        test_graph_dict[num].ndata['features'] = features[support_index + test_index_dict[num]]

    for num in test:
        test_mask_dict[num] = torch.zeros(test_graph_dict[num].number_of_nodes(), dtype=torch.bool)
        test_mask_dict[num][test_nid_dict[num]] = 1
        test_nid_dict[num] = torch.tensor(test_nid_dict[num], dtype=torch.int64)
        # normalize weight & add self-loop
        normalize_weight(test_graph_dict[num])
        test_graph_dict[num].add_edges(
            test_graph_dict[num].nodes(), test_graph_dict[num].nodes(),
            {'weight': torch.ones(test_graph_dict[num].number_of_nodes(), dtype=torch.float).unsqueeze(1)})

    test_dict = {
        'graph': test_graph_dict,
        'nid': test_nid_dict,
        'mask': test_mask_dict,
        'origin_id': test_cell_origin_id_dict
    }
    time_used = time.time() - tic

    if params.score:
        return total_cell, num_genes, num_labels, id2label, test_dict, test_label_dict, time_used
    else:
        return total_cell, num_genes, num_labels, id2label, test_dict, time_used


def get_id_2_gene(species_data_path, species, tissue, filetype):
    data_path = species_data_path
    data_files = list(data_path.glob(f'{species}_{tissue}*_data.{filetype}'))
    if len(data_files) < 1:
        raise FileNotFoundError(f"Missing data files {data_path}/{species}_{tissue}*_data.{filetype}")

    genes = set()
    for file in data_files:
        if filetype == 'csv':
            data = pd.read_csv(file, dtype=np.str, header=0).values[:, 0]
        else:
            data = pd.read_csv(file, compression='gzip', header=0).values[:, 0]
        genes = genes.union(set(data))
    return sorted(genes)


def get_id_2_label_and_label_statistics(species_data_path, species, tissue):
    data_path = species_data_path
    cell_files = data_path.glob(f'{species}_{tissue}*_celltype.csv')
    cell_types = set()
    cell_type_list = list()
    for file in cell_files:
        df = pd.read_csv(file, dtype=np.str, header=0)
        df['Cell_type'] = df['Cell_type'].map(str.strip)
        cell_types = set(df.values[:, 2]) | cell_types
        cell_type_list.extend(df.values[:, 2].tolist())
    id2label = list(cell_types)
    label_statistics = dict(collections.Counter(cell_type_list))
    return id2label, label_statistics


def save_statistics(statistics_path, id2label, id2gene, tissue):
    gene_path = statistics_path / f'{tissue}_genes.txt'
    label_path = statistics_path / f'{tissue}_cell_type.txt'
    with open(gene_path, 'w', encoding='utf-8') as f:
        for gene in id2gene:
            f.write(gene + '\r\n')
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in id2label:
            f.write(label + '\r\n')


def load_annotation_data_internal(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    species = params.species
    tissue = params.tissue

    proj_path = Path(params.proj_path)
    species_data_path = proj_path / 'train' / species
    graph_path = proj_path / 'pretrained' / species / 'graphs'
    statistics_path = proj_path / 'pretrained' / species / 'statistics'

    if not species_data_path.exists():
        raise NotImplementedError

    if not statistics_path.exists():
        statistics_path.mkdir(parents=True)
    if not graph_path.exists():
        graph_path.mkdir(parents=True)

    # generate gene statistics file
    id2gene = get_id_2_gene(species_data_path, species, tissue, filetype=params.filetype)
    # generate cell label statistics file
    id2label, label_statistics = get_id_2_label_and_label_statistics(species_data_path, species, tissue)
    total_cell = sum(label_statistics.values())
    for label, num in label_statistics.items():
        if num / total_cell <= params.exclude_rate:
            id2label.remove(label)  # remove exclusive labels
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    save_statistics(statistics_path, id2label, id2gene, tissue)
    print(f"The build graph contains {num_genes} genes with {num_labels} labels supported.")

    graph = dgl.DGLGraph()
    gene_ids = torch.arange(num_genes, dtype=torch.int32).unsqueeze(-1)
    graph.add_nodes(num_genes, {'id': gene_ids})

    all_labels = []
    matrices = []
    num_cells = 0

    data_path = species_data_path
    data_files = data_path.glob(f'*{params.species}_{tissue}*_data.{params.filetype}')
    for data_file in data_files:
        number = ''.join(list(filter(str.isdigit, data_file.name)))
        type_file = species_data_path / f'{params.species}_{tissue}{number}_celltype.csv'

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_file, index_col=0)
        cell2type.columns = ['cell', 'type']
        cell2type['type'] = cell2type['type'].map(str.strip)
        cell2type['id'] = cell2type['type'].map(label2id)
        # filter out cells not in label-text
        filter_cell = np.where(~pd.isnull(cell2type['id']))[0]
        cell2type = cell2type.iloc[filter_cell]

        assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
        all_labels += cell2type['id'].tolist()

        if params.filetype not in ["csv", "gz"]:
            print(f'Not supported type for {data_path}. Please verify your data file')
            continue

        # load data file then update graph
        df = pd.read_csv(data_file, index_col=0).transpose(copy=True)  # (cell, gene)

        # filter out cells not in label-text
        df = df.iloc[filter_cell]
        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)

        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(f"{params.species}_{tissue}{num}_data.{params.filetype} -> "
              f"Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%")

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        cell_idx = row_idx + graph.number_of_nodes()  # cell_index
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        num_cells += len(df)

        ids = torch.tensor([-1] * len(df), dtype=torch.int32).unsqueeze(-1)
        graph.add_nodes(len(df), {'id': ids})
        graph.add_edges(cell_idx, gene_idx, {'weight': torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)})
        graph.add_edges(gene_idx, cell_idx, {'weight': torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        print(f'#Nodes in Graph: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')

    assert len(all_labels) == num_cells

    save_npz(graph_path / f'{params.species}_{tissue}_data', vstack(matrices))

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    assert sparse_feat.shape[0] == num_cells
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat.T)
    gene_feat = gene_pca.transform(sparse_feat.T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    print('------Train label statistics------')

    for i, label in enumerate(id2label, start=1):
        print(f"#{i} [{label}]: {label_statistics[label]}")

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)
    graph.ndata['features'] = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
    labels = torch.tensor([-1] * num_genes + all_labels, dtype=torch.long)  # [gene_num+train_num]
    # split train set and test set
    per = np.random.permutation(range(num_genes, num_genes + num_cells))
    test_ids = torch.tensor(per[:int(num_cells // ((1 - params.test_rate) / params.test_rate + 1))])
    train_ids = torch.tensor(per[int(num_cells // ((1 - params.test_rate) / params.test_rate + 1)):])
    # normalize weight

    # normalize weight
    normalize_weight(graph)

    # add self-loop
    graph.add_edges(graph.nodes(), graph.nodes(),
                    {'weight': torch.ones(graph.number_of_nodes(), dtype=torch.float).unsqueeze(1)})

    return num_cells, num_genes, num_labels, graph, train_ids, test_ids, labels


########################################
# Cell Type Annotation for SVM
###########################################


def get_id_2_gene_svm(gene_statistics_path, train_dir, tissue):
    if not gene_statistics_path.exists():
        data_files = Path(train_dir).glob(f"*{tissue}*_data.csv")
        genes = None
        for file in data_files:
            data = pd.read_csv(file, dtype=np.str, header=0).values[:, 0]
            if genes is None:
                genes = set(data)
            else:
                genes = genes | set(data)
        id2gene = list(genes)
        id2gene.sort()
        with open(gene_statistics_path, "w", encoding="utf-8") as f:
            for gene in id2gene:
                f.write(gene + "\r\n")
    else:
        id2gene = []
        with open(gene_statistics_path, encoding="utf-8") as f:
            for line in f:
                id2gene.append(line.strip())
    return id2gene


def get_id_2_label_svm(cell_statistics_path, train_dir, tissue):
    if not cell_statistics_path.exists():
        data_path = Path(train_dir)
        cell_files = data_path.glob(f"*{tissue}*_celltype.csv")
        cell_types = set()
        for file in cell_files:
            df = pd.read_csv(file, dtype=np.str, header=0)
            df["Cell_type"] = df["Cell_type"].map(str.strip)
            cell_types = set(df.values[:, 2]) | cell_types
            # cell_types = set(pd.read_csv(file, dtype=np.str, header=0).values[:, 2]) | cell_types
        id2label = list(cell_types)
        with open(cell_statistics_path, "w", encoding="utf-8") as f:
            for cell_type in id2label:
                f.write(cell_type + "\r\n")
    else:
        id2label = []
        with open(cell_statistics_path, encoding="utf-8") as f:
            for line in f:
                id2label.append(line.strip())
    return id2label


def load_svm_data(params):
    random_seed = params.random_seed
    train = params.train_dataset
    test = params.test_dataset
    tissue = params.tissue
    species = params.species
    dense_dim = params.dense_dim
    statistics_path = Path(params.statistics_path)
    map_path = Path(params.map_path) / species

    map_dict = get_map_dict(map_path, tissue)

    gene_statistics_path = statistics_path / (tissue + "_genes.txt")  # train+test gene
    cell_statistics_path = statistics_path / (tissue + "_cell_type.txt")  # train labels

    # generate gene statistics file
    id2gene = get_id_2_gene_svm(gene_statistics_path, os.path.join(params.train_dir, species), tissue)
    # generate cell label statistics file
    id2label = get_id_2_label_svm(cell_statistics_path, os.path.join(params.train_dir, species), tissue)

    train_num, test_num = 0, 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"totally {num_genes} genes, {num_labels} labels.")

    train_labels = []
    test_label_dict = dict()  # test label dict
    test_index_dict = dict()  # test-num: [begin-index, end-index]
    test_cell_id_dict = dict()  # test-num: ["c1", "c2"...]
    # TODO
    matrices = []
    print(train, test)
    for num in train + test:
        start = time.time()
        if num in train:
            data_path = os.path.join(params.train_dir, species, params.species + "_" + tissue + str(num) + "_data.csv")
            type_path = os.path.join(params.train_dir, species,
                                     params.species + "_" + tissue + str(num) + "_celltype.csv")
        else:
            data_path = os.path.join(params.test_dir, species, params.species + "_" + tissue + str(num) + "_data.csv")
            type_path = os.path.join(params.test_dir, species,
                                     params.species + "_" + tissue + str(num) + "_celltype.csv")

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_path, index_col=0)
        cell2type.columns = ["cell", "type"]
        cell2type["type"] = cell2type["type"].map(str.strip)
        if num in train:
            cell2type["id"] = cell2type["type"].map(label2id)
            assert not cell2type["id"].isnull().any(), "something wrong in celltype file."
            train_labels += cell2type["id"].tolist()
        else:
            # test_labels += cell2type["type"].tolist()
            test_label_dict[num] = cell2type["type"].tolist()

        # load data file then update graph
        df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        if num in test:
            test_cell_id_dict[num] = list(df.columns)
        df = df.transpose(copy=True)  # (cell, gene)

        assert cell2type["cell"].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]
        print(f"Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%")
        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values

        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        if num in train:
            train_num += len(df)
        else:
            test_index_dict[num] = list(range(train_num + test_num, train_num + test_num + len(df)))
            test_num += len(df)
        print(f"Costs {time.time() - start:.3f} s in total.")
    train_labels = np.array(list(map(int, train_labels)))

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    test_feat_dict = dict()
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:train_num].T)
    gene_feat = gene_pca.transform(sparse_feat[:train_num].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f"[PCA] Gene EVR: {gene_evr:.2f} %.")

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)  # [total_cell_num, d]
    train_cell_feat = cell_feat[:train_num]

    for num in test_label_dict.keys():
        test_feat_dict[num] = cell_feat[test_index_dict[num]]

    return (num_labels, train_labels, train_cell_feat, map_dict, np.array(id2label, dtype=np.str), test_label_dict,
            test_feat_dict, test_cell_id_dict)


#######################################################
#For Single Cell ACTINN
#######################################################


# Get common genes, normalize  and scale the sets
def scale_sets(sets, normalize=True):
    # input -- a list of all the sets to be scaled
    # output -- scaled sets
    # normalize -- Skip library size + log normalize if set to False (scDeepsort data prenormalized)
    common_genes = set(sets[0].index)
    for i in range(1, len(sets)):
        common_genes = set.intersection(set(sets[i].index), common_genes)
    common_genes = sorted(list(common_genes))
    sep_point = [0]
    for i in range(len(sets)):
        sets[i] = sets[i].loc[common_genes, ]
        sep_point.append(sets[i].shape[1])
    total_set = np.array(pd.concat(sets, axis=1, sort=False), dtype=np.float32)
    if normalize:
        total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
        total_set = np.log2(total_set + 1)
        expr = np.sum(total_set, axis=1)
        total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)), ]
        cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
        total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)), ]
    for i in range(len(sets)):
        sets[i] = total_set[:, sum(sep_point[:(i + 1)]):sum(sep_point[:(i + 2)])]
    return sets


# Turn labels into matrix
def one_hot_matrix(labels, C):
    # input -- labels (true labels of the sets), C (# types)
    # output -- one hot matrix with shape (# types, # samples)
    labels = torch.tensor(labels)
    C = torch.tensor(C)
    one_hot_matrix = F.one_hot(labels, C)
    return one_hot_matrix.T


# Make types to labels dictionary
def type_to_label_dict(types):
    # input -- types
    # output -- type_to_label dictionary
    type_to_label_dict = {}
    all_type = sorted(set(types))
    for i in range(len(all_type)):
        type_to_label_dict[all_type[i]] = i
    return type_to_label_dict


# Convert types to labels
def convert_type_to_label(types, type_to_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    types = list(types)
    labels = list()
    for type in types:
        labels.append(type_to_label_dict[type])
    return labels


# Function to create placeholders
def create_placeholders(n_x, n_y):
    X = torch.zeros(n_x)
    Y = torch.zeros(n_y)
    return X, Y


def load_actinn_data(train_data_paths: List[str], train_label_paths: List[str], test_data_path: str,
                     test_label_path: str, normalize: bool = False):
    # TODO: multiple test datasets
    train_set_dfs = []
    train_label_dfs = []
    for train_data_path, train_label_path in zip(train_data_paths, train_label_paths):
        train_set_df = pd.read_csv(train_data_path, index_col=0)
        train_set_df.index = train_set_df.index.str.upper()
        train_set_df = train_set_df.loc[~train_set_df.index.duplicated(keep="first")]
        train_label_df = pd.read_csv(train_label_path, index_col=0)

        train_set_dfs.append(train_set_df)
        train_label_dfs.append(train_label_df)

    train_set = pd.concat(train_set_dfs, axis=1, join="inner")
    train_label = pd.concat(train_label_dfs, axis=0)

    test_set = pd.read_csv(test_data_path, index_col=0)
    test_set.index = test_set.index.str.upper()
    test_set = test_set.loc[~test_set.index.duplicated(keep="first")]
    test_label = pd.read_csv(test_label_path, index_col=0)

    nt = train_label.iloc[:, 1].unique().size
    train_set, test_set = scale_sets([train_set, test_set], normalize=normalize)
    type_to_label_dict_out = type_to_label_dict(train_label.iloc[:, 1])
    label_to_type_dict = {v: k for k, v in type_to_label_dict_out.items()}
    print(f"Cell Types in training set:")
    pprint.pprint(type_to_label_dict_out)
    train_label = convert_type_to_label(train_label.iloc[:, 1], type_to_label_dict_out)
    train_label = one_hot_matrix(train_label, nt)
    print(f"# Trainng cells: {train_label.shape[1]:,}")

    total_test_cells = test_label.shape[0]
    indicator = test_label.iloc[:, 1].isin(type_to_label_dict_out)
    test_label = test_label[indicator]
    barcode = test_label.iloc[:, 0].tolist()
    test_set = test_set[:, indicator]
    test_label = convert_type_to_label(test_label.iloc[:, 1], type_to_label_dict_out)
    test_label = one_hot_matrix(test_label, nt)
    print(f"# Testing cells {test_label.shape[1]:,} (original number of cells = {total_test_cells:,})")

    # Convert to train_set and test_set to tensor
    train_set = torch.from_numpy(train_set)
    test_set = torch.from_numpy(test_set)

    return train_set, train_label, test_set, test_label, barcode, label_to_type_dict


#######################################################
#For Celltypist Model
#######################################################
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
_samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")


def get_sample_data_celltypist(filename: str) -> str:
    """Get the full path to the sample input data included in the package."""
    return os.path.join(_samples_path, filename)


def get_sample_csv_celltypist() -> str:
    """
    Get the full path to the sample csv file included in the package.
    Returns
    ----------
    str
        A string of the full path to the sample csv file (`sample_cell_by_gene.csv`).
    """
    return _get_sample_data("sample_cell_by_gene.csv")


def downsample_adata(adata: AnnData, mode: str = 'total', n_cells: Optional[int] = None, by: Optional[str] = None,
                     balance_cell_type: bool = False, random_state: int = 0,
                     return_index: bool = True) -> Union[AnnData, np.ndarray]:
    """
    Downsample cells to a given number (either in total or per cell type).
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object representing the input data.
    mode
        The way downsampling is performed. Default to downsampling the input cells to a total of `n_cells`.
        Set to `'each'` if you want to downsample cells within each cell type to `n_cells`.
        (Default: `'total'`)
    n_cells
        The total number of cells (`mode = 'total'`) or the number of cells from each cell type (`mode = 'each'`) to sample.
        For the latter, all cells from a given cell type will be selected if its cell number is fewer than `n_cells`.
    by
        Key (column name) of the input AnnData representing the cell types.
    balance_cell_type
        Whether to balance the cell type frequencies when `mode = 'total'`.
        Setting to `True` will sample rare cell types with a higher probability, ensuring close-to-even cell type compositions.
        This argument is ignored if `mode = 'each'`.
        (Default: `False`)
    random_state
        Random seed for reproducibility.
    return_index
        Only return the downsampled cell indices.
        Setting to `False` if you want to get a downsampled version of the input AnnData.
        (Default: `True`)
    Returns
    ----------
    Depending on `return_index`, returns the downsampled cell indices or a subset of the input AnnData.
    """
    np.random.seed(random_state)
    if n_cells is None:
        raise ValueError(f"?? Please provide `n_cells`")
    if mode == 'total':
        if n_cells >= adata.n_obs:
            raise ValueError(f"?? `n_cells` ({n_cells}) should be fewer than the total number of cells ({adata.n_obs})")
        if balance_cell_type:
            if by is None:
                raise KeyError(
                    f"?? Please specify the cell type column if you want to balance the cell type frequencies")
            labels = adata.obs[by]
            celltype_freq = np.unique(labels, return_counts=True)
            len_celltype = len(celltype_freq[0])
            mapping = pd.Series(1 / (celltype_freq[1] * len_celltype), index=celltype_freq[0])
            p = mapping[labels].values
            sampled_cell_index = np.random.choice(adata.n_obs, n_cells, replace=False, p=p)
        else:
            sampled_cell_index = np.random.choice(adata.n_obs, n_cells, replace=False)
    elif mode == 'each':
        if by is None:
            raise KeyError(f"?? Please specify the cell type column for downsampling")
        celltypes = np.unique(adata.obs[by])
        sampled_cell_index = np.concatenate([
            np.random.choice(
                np.where(adata.obs[by] == celltype)[0], min([n_cells, np.sum(adata.obs[by] == celltype)]),
                replace=False) for celltype in celltypes
        ])
    else:
        raise ValueError(f"?? Unrecognized `mode` value, should be one of `'total'` or `'each'`")
    if return_index:
        return sampled_cell_index
    else:
        return adata[sampled_cell_index].copy()


def to_vector_celltypist(_vector_or_file):

    if isinstance(_vector_or_file, str):
        try:
            return pd.read_csv(_vector_or_file, header=None)[0].values
        except Exception as e:
            raise Exception(f"?? {e}")
    else:
        return _vector_or_file


def to_array_celltypist(_array_like):

    if isinstance(_array_like, pd.DataFrame):
        return _array_like.values
    elif isinstance(_array_like, spmatrix):
        return _array_like.toarray()
    elif isinstance(_array_like, np.matrix):
        return np.array(_array_like)
    elif isinstance(_array_like, np.ndarray):
        return _array_like
    else:
        raise ValueError(f"?? Please provide a valid array-like object as input")


def prepare_data_celltypist(X, labels, genes, transpose):

    if (X is None) or (labels is None):
        raise Exception("?? Missing training data and/or training labels. Please provide both arguments")
    if isinstance(X, AnnData) or (isinstance(X, str) and X.endswith('.h5ad')):
        adata = sc.read(X) if isinstance(X, str) else X
        adata.var_names_make_unique()
        if adata.X.min() < 0:
            logger.info("?? Detected scaled expression in the input data, will try the .raw attribute")
            try:
                indata = adata.raw.X
                genes = adata.raw.var_names
            except Exception as e:
                raise Exception(f"?? Fail to use the .raw attribute in the input object. {e}")
        else:
            indata = adata.X
            genes = adata.var_names
        if isinstance(labels, str) and (labels in adata.obs):
            labels = adata.obs[labels]
        else:
            labels = to_vector_celltypist(labels)
    elif isinstance(X, str) and X.endswith(('.csv', '.txt', '.tsv', '.tab', '.mtx', '.mtx.gz')):
        adata = sc.read(X)
        if transpose:
            adata = adata.transpose()
        if X.endswith(('.mtx', '.mtx.gz')):
            if genes is None:
                raise Exception("?? Missing `genes`. Please provide this argument together with the input mtx file")
            genes = to_vector_celltypist(genes)
            if len(genes) != adata.n_vars:
                raise ValueError(f"?? The number of genes provided does not match the number of genes in {X}")
            adata.var_names = np.array(genes)
        adata.var_names_make_unique()
        if not float(adata.X.max()).is_integer():
            logger.warn(f"?? Warning: the input file seems not a raw count matrix. The trained model may be biased")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        indata = adata.X
        genes = adata.var_names
        labels = to_vector_celltypist(labels)
    elif isinstance(X, str):
        raise ValueError("?? Invalid input. Supported types: .csv, .txt, .tsv, .tab, .mtx, .mtx.gz and .h5ad")
    else:
        logger.info("?? The input training data is processed as an array-like object")
        indata = X
        if transpose:
            indata = indata.transpose()
        if isinstance(indata, pd.DataFrame):
            genes = indata.columns
        else:
            if genes is None:
                raise Exception(
                    "?? Missing `genes`. Please provide this argument together with the input training data")
            genes = to_vector_celltypist(genes)
        labels = to_vector_celltypist(labels)
    return indata, labels, genes


def LRClassifier_celltypist(indata, labels, C, solver, max_iter, n_jobs, **kwargs) -> LogisticRegression:
    """For internal use.

    Get the logistic Classifier.

    """
    no_cells = len(labels)
    if solver is None:
        solver = 'sag' if no_cells > 50000 else 'lbfgs'
    elif solver not in ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'):
        raise ValueError(
            f"?? Invalid `solver`, should be one of `'liblinear'`, `'lbfgs'`, `'newton-cg'`, `'sag'`, and `'saga'`")
    logger.info(f"??? Training data using logistic regression")
    if (no_cells > 100000) and (indata.shape[1] > 10000):
        logger.warn(
            f"?? Warning: it may take a long time to train this dataset with {no_cells} cells and {indata.shape[1]} genes, try to downsample cells and/or restrict genes to a subset (e.g., hvgs)"
        )
    print("LRClassifier training start...")
    classifier = LogisticRegression(C=C, solver=solver, max_iter=max_iter, multi_class='ovr', n_jobs=n_jobs, **kwargs)
    classifier.fit(indata, labels)
    return classifier


def SGDClassifier_celltypist(indata, labels, alpha, max_iter, n_jobs, mini_batch, batch_number, batch_size, epochs,
                             balance_cell_type, **kwargs) -> SGDClassifier:
    """For internal use.

    Get the SGDClassifier.

    """
    classifier = SGDClassifier(loss='log', alpha=alpha, max_iter=max_iter, n_jobs=n_jobs, **kwargs)
    if not mini_batch:
        logger.info(f"??? Training data using SGD logistic regression")
        if (len(labels) > 100000) and (indata.shape[1] > 10000):
            logger.warn(
                f"?? Warning: it may take a long time to train this dataset with {len(labels)} cells and {indata.shape[1]} genes, try to downsample cells and/or restrict genes to a subset (e.g., hvgs)"
            )
        print("SGDlassifier training start...")
        classifier.fit(indata, labels)
    else:
        logger.info(f"??? Training data using mini-batch SGD logistic regression")
        no_cells = len(labels)
        if no_cells < 10000:
            logger.warn(
                f"?? Warning: the number of cells ({no_cells}) is not big enough to conduct a proper mini-batch training. You may consider using traditional SGD classifier (mini_batch = False)"
            )
        if no_cells <= batch_size:
            raise ValueError(
                f"?? Number of cells ({no_cells}) is fewer than the batch size ({batch_size}). Decrease `batch_size`, or use SGD directly (mini_batch = False)"
            )
        no_cells_sample = min([batch_number * batch_size, no_cells])
        starts = np.arange(0, no_cells_sample, batch_size)
        if balance_cell_type:
            celltype_freq = np.unique(labels, return_counts=True)
            len_celltype = len(celltype_freq[0])
            mapping = pd.Series(1 / (celltype_freq[1] * len_celltype), index=celltype_freq[0])
            p = mapping[labels].values
        for epoch in range(1, (epochs + 1)):
            logger.info(f"? Epochs: [{epoch}/{epochs}]")
            if not balance_cell_type:
                sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace=False)
            else:
                sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace=False, p=p)
            for start in starts:
                print("SGDlassifier training start...")
                classifier.partial_fit(indata[sampled_cell_index[start:start + batch_size]],
                                       labels[sampled_cell_index[start:start + batch_size]], classes=np.unique(labels))
    return classifier


#######################################################
#For singlecellnet
#######################################################


def ctMerge(sampTab, annCol, ctVect, newName):
    oldRows = np.isin(sampTab[annCol], ctVect)
    newSampTab = sampTab.copy()
    newSampTab.loc[oldRows, annCol] = newName
    return newSampTab


def ctRename(sampTab, annCol, oldName, newName):
    oldRows = sampTab[annCol] == oldName
    newSampTab = sampTab.copy()
    newSampTab.loc[oldRows, annCol] = newName
    return newSampTab


def splitCommonAnnData(adata, ncells, dLevel="cell_ontology_class", cellid=None, cells_reserved=3):
    if cellid == None:
        adata.obs[cellid] = adata.obs.index
    cts = set(adata.obs[dLevel])

    trainingids = np.empty(0)
    for ct in cts:
        print(ct, ": ")
        aX = adata[adata.obs[dLevel] == ct, :]
        ccount = aX.n_obs - cells_reserved
        ccount = min([ccount, ncells])
        print(aX.n_obs)
        trainingids = np.append(trainingids, np.random.choice(aX.obs[cellid].values, ccount, replace=False))

    val_ids = np.setdiff1d(adata.obs[cellid].values, trainingids, assume_unique=True)
    aTrain = adata[np.isin(adata.obs[cellid], trainingids, assume_unique=True), :]
    aTest = adata[np.isin(adata.obs[cellid], val_ids, assume_unique=True), :]
    return ([aTrain, aTest])


def splitCommon(expData, ncells, sampTab, dLevel="cell_ontology_class", cells_reserved=3):
    cts = set(sampTab[dLevel])
    trainingids = np.empty(0)
    for ct in cts:
        aX = expData.loc[sampTab[dLevel] == ct, :]
        print(ct, ": ")
        ccount = len(aX.index) - cells_reserved
        ccount = min([ccount, ncells])
        print(ccount)
        trainingids = np.append(trainingids, np.random.choice(aX.index.values, ccount, replace=False))
    val_ids = np.setdiff1d(sampTab.index, trainingids, assume_unique=True)
    aTrain = expData.loc[np.isin(sampTab.index.values, trainingids, assume_unique=True), :]
    aTest = expData.loc[np.isin(sampTab.index.values, val_ids, assume_unique=True), :]
    return ([aTrain, aTest])


def annSetUp(species="mmusculus"):
    annot = sc.queries.biomart_annotations(
        species,
        ["external_gene_name", "go_id"],
    )
    return annot


def getGenesFromGO(GOID, annList):
    if (str(type(GOID)) != "<class 'str'>"):
        return annList.loc[annList.go_id.isin(GOID), :].external_gene_name.sort_values().to_numpy()
    else:
        return annList.loc[annList.go_id == GOID, :].external_gene_name.sort_values().to_numpy()


def dumbfunc(aNamedList):
    return aNamedList.index.values


def GEP_makeMean(expDat, groupings, type='mean'):
    if (type == "mean"):
        return expDat.groupby(groupings).mean()
    if (type == "median"):
        return expDat.groupby(groupings).median()


def utils_myDist(expData):
    numSamps = len(expData.index)
    result = np.subtract(np.ones([numSamps, numSamps]), expData.T.corr())
    del result.index.name
    del result.columns.name
    return result


def utils_stripwhite(string):
    return string.strip()


def utils_myDate():
    d = datetime.datetime.today()
    return d.strftime("%b_%d_%Y")


def utils_strip_fname(string):
    sp = string.split("/")
    return sp[len(sp) - 1]


def utils_stderr(x):
    return (stats.sem(x))


def zscore(x, meanVal, sdVal):
    return np.subtract(x, meanVal) / sdVal


def zscoreVect(genes, expDat, tVals, ctt, cttVec):
    res = {}
    x = expDat.loc[cttVec == ctt, :]
    for gene in genes:
        xvals = x[gene]
        res[gene] = pd.series(data=zscore(xvals, tVals[ctt]['mean'][gene], tVals[ctt]['sd'][gene]),
                              index=xvals.index.values)
    return res


def downSampleW(vector, total=1e5, dThresh=0):
    vSum = np.sum(vector)
    dVector = total / vSum
    res = dVector * vector
    res[res < dThresh] = 0
    return res


def weighted_down(expDat, total, dThresh=0):
    rSums = expDat.sum(axis=1)
    dVector = np.divide(total, rSums)
    res = expDat.mul(dVector, axis=0)
    res[res < dThresh] = 0
    return res


def trans_prop(expDat, total, dThresh=0):
    rSums = expDat.sum(axis=1)
    dVector = np.divide(total, rSums)
    res = expDat.mul(dVector, axis=0)
    res[res < dThresh] = 0
    return np.log(res + 1)


def trans_zscore_col(expDat):
    return expDat.apply(stats.zscore, axis=0)


def trans_zscore_row(expDat):
    return expDat.T.apply(stats.zscore, axis=0).T


def trans_binarize(expData, threshold=1):
    expData[expData < threshold] = 0
    expData[expData > 0] = 1
    return expData


def getUniqueGenes(genes, transID='id', geneID='symbol'):
    genes2 = genes.copy()
    genes2.index = genes2[transID]
    genes2.drop_duplicates(subset=geneID, inplace=True, keep="first")
    del genes2.index.name
    return genes2


def removeRed(expData, genes, transID="id", geneID="symbol"):
    genes2 = getUniqueGenes(genes, transID, geneID)
    return expData.loc[:, genes2.index.values]


def cn_correctZmat_col(zmat):

    def myfuncInf(vector):
        mx = np.max(vector[vector < np.inf])
        mn = np.min(vector[vector > (np.inf * -1)])
        res = vector.copy()
        res[res > mx] = mx
        res[res < mn] = mn
        return res

    return zmat.apply(myfuncInf, axis=0)


def cn_correctZmat_row(zmat):

    def myfuncInf(vector):
        mx = np.max(vector[vector < np.inf])
        mn = np.min(vector[vector > (np.inf * -1)])
        res = vector.copy()
        res[res > mx] = mx
        res[res < mn] = mn
        return res

    return zmat.apply(myfuncInf, axis=1)


def makeExpMat(adata):
    expMat = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    return expMat


def makeSampTab(adata):
    sampTab = adata.obs
    return sampTab


def sc_statTab(expDat, dThresh=0):
    geneNames = expDat.columns.values
    muAll = sc_compMu(expDat, threshold=dThresh)
    alphaAll = sc_compAlpha(expDat, threshold=dThresh)
    meanAll = expDat.apply(np.mean, axis=0)
    covAll = expDat.apply(sc_cov, axis=0)
    fanoAll = expDat.apply(sc_fano, axis=0)
    maxAll = expDat.apply(np.max, axis=0)
    sdAll = expDat.apply(np.std, axis=0)
    statTabAll = pd.concat([muAll, alphaAll, meanAll, covAll, fanoAll, maxAll, sdAll], axis=1)
    statTabAll.columns = ["mu", "alpha", "overall_mean", "cov", "fano", "max_val", "sd"]
    return statTabAll


def sc_compAlpha(expDat, threshold=0, pseudo=False):

    def singleGene(col, thresh, pseu):
        if pseudo:
            return (np.sum(col > thresh) + 1) / float(len(col) + 1)
        else:
            return np.sum(col > thresh) / float(len(col))

    return expDat.apply(singleGene, axis=0, args=(
        threshold,
        pseudo,
    ))


def sc_compMu(
    expDat,
    threshold=0,
):

    def singleGene(col, thresh):
        return np.sum(col[col > thresh]) / float(len(col[col > thresh]))

    return expDat.apply(singleGene, axis=0, args=(threshold, )).fillna(0)


def repNA(df):
    return df.fillna(0)


def sc_fano(vector):
    return np.true_divide(np.var(vector), np.mean(vector))


def sc_cov(vector):
    return np.true_divide(np.std(vector), np.mean(vector))


def sc_filterGenes(geneStats, alpha1=0.1, alpha2=0.01, mu=2):
    return geneStats[np.logical_or(geneStats.alpha > alpha1, np.logical_and(geneStats.alpha > alpha2,
                                                                            geneStats.mu > mu))].index.values


def sc_filterCells(sampTab, minVal=1e3, maxValQuant=0.95):
    q = np.quantile(sampTab.umis, maxValQuant)
    return sampTab[np.logical_and(sampTab.umis > minVal, sampTab.umis < q)].index.values


def sc_findEnr(expDat, sampTab, dLevel="group"):
    summ = expDat.groupby(sampTab[dLevel]).median()
    dict = {}
    for n in range(0, summ.index.size):
        temp = np.subtract(summ.iloc[n, :], summ.drop(index=summ.index.values[n]).apply(np.median, axis=0))
        dict[summ.index.values[n]] = summ.columns.values[np.argsort(-1 * temp)].tolist()
    return dict


def enrDiff(expDat, sampTab, dLevel="group"):
    groups = np.unique(sampTab[dLevel])
    summ = expDat.groupby(sampTab[dLevel]).median()
    ref = summ.copy()
    for n in range(0, ref.index.size):
        summ.iloc[n, :] = np.subtract(summ.iloc[n, :], ref.drop(index=ref.index.values[n]).apply(np.median, axis=0))
    return summ


def binGenesAlpha(geneStats, nbins=20):
    max = np.max(geneStats['alpha'])
    min = np.min(geneStats['alpha'])
    rrange = max - min
    inc = rrange / nbins
    threshs = np.arange(max, min, -1 * inc)
    res = pd.DataFrame(index=geneStats.index.values, data=np.arange(0, geneStats.index.size, 1), columns=["bin"])
    for i in range(0, len(threshs)):
        res.loc[geneStats["alpha"] <= threshs[i], 0] = len(threshs) - i
    return res


def binGenes(geneStats, nbins=20, meanType="overall_mean"):
    max = np.max(geneStats[meanType])
    min = np.min(geneStats[meanType])
    rrange = max - min
    inc = rrange / nbins
    threshs = np.arange(max, min, -1 * inc)
    res = pd.DataFrame(index=geneStats.index.values, data=np.arange(0, geneStats.index.size, 1), columns=["bin"])
    for i in range(0, len(threshs)):
        res.loc[geneStats[meanType] <= threshs[i], "bin"] = len(threshs) - i
    return res


def findVarGenes(geneStats, zThresh=2, meanType="overall_mean"):
    zscs = pd.DataFrame(index=geneStats.index.values, data=np.zeros([geneStats.index.size, 3]),
                        columns=["alpha", meanType, "mu"])
    mTypes = ["alpha", meanType, "mu"]
    scaleVar = ["fano", "fano", "cov"]
    for i in range(0, 3):
        sg = binGenes(geneStats, meanType=mTypes[i])
        bbins = np.unique(sg["bin"])
        for b in bbins:
            if (np.unique(geneStats.loc[sg.bin == b, scaleVar[i]]).size > 1):
                tmpZ = stats.zscore(geneStats.loc[sg.bin == b, scaleVar[i]])
            else:
                tmpZ = np.zeros(geneStats.loc[sg.bin == b, scaleVar[i]].index.size).T
            zscs.loc[sg.bin == b, mTypes[i]] = tmpZ
    return (zscs.loc[np.logical_and(zscs.iloc[:, 0] > zThresh,
                                    np.logical_and(zscs.iloc[:, 1] > zThresh, zscs.iloc[:, 2] > zThresh))].index.values)


def sc_sampR_to_pattern(sampR):
    d_ids = np.unique(sampR)
    nnnc = len(sampR)
    dict = {}
    for d_id in d_ids:
        x = np.zeros(nnnc)
        x[np.where(np.isin(sampR, d_id))] = 1
        dict[d_id] = x
    return dict


def minTab(sampTab, dLevel):
    myMin = np.min(sampTab[dLevel].value_counts())
    grouped = sampTab.groupby(dLevel, as_index=False)
    res = grouped.apply(lambda x: x.sample(n=myMin, replace=False)).reset_index(level=0, drop=True)
    return res


def sc_testPattern(pattern, expDat):
    yy = np.vstack([pattern, np.ones(len(pattern))]).T
    p, _, _, _ = np.linalg.lstsq(yy, expDat)
    n = len(expDat)
    k = len(p)
    sigma2 = np.sum((expDat - np.dot(yy, p))**2) / (n - k)  # RMSE
    C_pre = np.diag(np.linalg.inv(np.dot(yy.T, yy)))[0]  # covariance matrix
    C = np.sqrt(sigma2 * C_pre)
    SS_tot = np.sum((expDat - np.mean(expDat))**2)
    SS_err = np.sum((np.dot(yy, p) - expDat)**2)
    Rsq = 1 - SS_err / SS_tot
    t_val = np.divide(p[0], C)
    res = pd.DataFrame(index=expDat.columns)
    res["pval"] = 2 * stats.t.sf(np.abs(t_val), df=(n - k))
    res["cval"] = (Rsq**0.5) * np.sign(t_val)
    _, res["holm"], _, _ = smt.multipletests(res["pval"].values, method="holm")
    return res


def par_findSpecGenes(expDat, sampTab, dLevel="group", minSet=True):
    if minSet:
        samps = minTab(sampTab, dLevel)
    else:
        samps = sampTab.copy()
    pats = sc_sampR_to_pattern(samps[dLevel])
    exps = expDat.loc[samps.index, :]
    res = {}
    levels = list(pats.keys())
    for i in range(0, len(levels)):
        res[levels[i]] = sc_testPattern(pats[levels[i]], exps)
    return res


def getTopGenes(xDat, topN=3):
    return xDat.sort_values(by='cval', ascending=False).index.values[0:topN]


def getSpecGenes(xDatList, topN=50):
    groups = list(xDatList.keys())
    allG = []
    for i in range(0, len(groups)):
        topNs = getTopGenes(xDatList[groups[i]], topN)
        allG.append(topNs)
    allG = np.array(allG).reshape(-1, 1)
    u, c = np.unique(allG, return_counts=True)
    u[c > 1] = np.nan
    allG[~np.isin(allG, u)] = np.nan
    specGenes = allG.reshape(len(groups), topN)
    res = {}
    for i in range(0, len(specGenes)):
        res[groups[i]] = specGenes[i, ~pd.isnull(specGenes[i])].tolist()
    return res


def getTopGenesList(xDatList, topN=50):
    groups = list(xDatList.keys())
    temp = []
    for i in range(0, len(groups)):
        topNs = getTopGenes(xDatList[groups[i]], topN)
        res = ", ".join(topNs)
        temp.append(res)
    res = {}
    for i in range(0, len(groups)):
        res[groups[i]] = temp[i]
    return res


def csRenameOrth(adQuery, adTrain, orthTable, speciesQuery='human', speciesTrain='mouse'):
    _, _, cgenes = np.intersect1d(adQuery.var_names.values, orthTable[speciesQuery], return_indices=True)
    _, _, ccgenes = np.intersect1d(adTrain.var_names.values, orthTable[speciesTrain], return_indices=True)
    temp1 = np.zeros(len(orthTable.index.values), dtype=bool)
    temp2 = np.zeros(len(orthTable.index.values), dtype=bool)
    temp1[cgenes] = True
    temp2[ccgenes] = True
    common = np.logical_and(temp1, temp2)
    oTab = orthTable.loc[common.T, :]
    adT = adTrain[:, oTab[speciesTrain]]
    adQ = adQuery[:, oTab[speciesQuery]]
    adQ.var_names = adT.var_names
    return [adQ, adT]


def csRenameOrth2(expQuery, expTrain, orthTable, speciesQuery='human', speciesTrain='mouse'):
    _, _, cgenes = np.intersect1d(expQuery.columns.values, orthTable[speciesQuery], return_indices=True)
    _, _, ccgenes = np.intersect1d(expTrain.columns.values, orthTable[speciesTrain], return_indices=True)
    temp1 = np.zeros(len(orthTable.index.values), dtype=bool)
    temp2 = np.zeros(len(orthTable.index.values), dtype=bool)
    temp1[cgenes] = True
    temp2[ccgenes] = True
    common = np.logical_and(temp1, temp2)
    oTab = orthTable.loc[common.T, :]
    expT = expTrain.loc[:, oTab[speciesTrain]]
    expQ = expQuery.loc[:, oTab[speciesQuery]]
    expQ.columns = expT.columns
    return [expQ, expT]


def makePairTab(genes):
    pairs = list(combinations(genes, 2))
    labels = ['genes1', 'genes2']
    pTab = pd.DataFrame(data=pairs, columns=labels)
    pTab['gene_pairs'] = pTab['genes1'] + '_' + pTab['genes2']
    return (pTab)


def gnrAll(expDat, cellLabels):
    myPatternG = sc_sampR_to_pattern(cellLabels)
    res = {}
    groups = np.unique(cellLabels)
    for i in range(0, len(groups)):
        res[groups[i]] = sc_testPattern(myPatternG[groups[i]], expDat)
    return res


def getClassGenes(diffRes, topX=25, bottom=True):
    xi = ~pd.isna(diffRes["cval"])
    diffRes = diffRes.loc[xi, :]
    sortRes = diffRes.sort_values(by="cval", ascending=False)
    ans = sortRes.index.values[0:topX]
    if bottom:
        l = len(sortRes) - topX
        ans = np.append(ans, sortRes.index.values[l:]).flatten()
    return ans


def addRandToSampTab(classRes, sampTab, desc, id="cell_name"):
    cNames = classRes.index.values
    snames = sampTab.index.values
    rnames = np.setdiff1d(cNames, snames)
    stNew = pd.DataFrame()
    stNew["rid"] = rnames
    stNew["rdesc"] = "rand"
    stTop = sampTab[[id, desc]]
    stNew.columns = [id, desc]
    ans = stTop.append(stNew)
    return ans


def ptSmall(expMat, pTab):
    npairs = len(pTab.index)
    genes1 = pTab['genes1'].values
    genes2 = pTab['genes2'].values
    expTemp = expMat.loc[:, np.unique(np.concatenate([genes1, genes2]))]
    ans = pd.DataFrame(0, index=expTemp.index, columns=np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1 = expTemp.loc[:, genes1]
    temp2 = expTemp.loc[:, genes2]
    temp1.columns = np.arange(npairs)
    temp2.columns = np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = list(pTab[['gene_pairs']].values.T)
    return (ans)


def findBestPairs(xdiff, n=50, maxPer=3):
    xdiff = xdiff.sort_values(by=['cval'], ascending=False)
    genes = []
    genesTemp = list(xdiff.index.values)
    for g in genesTemp:
        genes.append(g[0].split("_"))
    genes = np.unique(np.array(genes).flatten())
    countList = dict(zip(genes, np.zeros(genes.shape)))
    i = 1
    ans = np.empty(0)
    xdiff_index = 0
    pair_names = xdiff.index.values
    while i < n:
        tmpAns = pair_names[xdiff_index]
        tgp = tmpAns[0].split('_')
        if countList[tgp[0]] < maxPer and countList[tgp[1]] < maxPer:
            ans = np.append(ans, tmpAns)
            countList[tgp[0]] = countList[tgp[0]] + 1
            countList[tgp[1]] = countList[tgp[1]] + 1
            i = i + 1
        xdiff_index = xdiff_index + 1
    return (np.array(ans))


def query_transform(expMat, genePairs):
    npairs = len(genePairs)
    ans = pd.DataFrame(0, index=expMat.index, columns=np.arange(npairs))
    genes1 = []
    genes2 = []
    for g in genePairs:
        sp = g.split("_")
        genes1.append(sp[0])
        genes2.append(sp[1])
    expTemp = expMat.loc[:, np.unique(np.concatenate([genes1, genes2]))]
    ans = pd.DataFrame(0, index=expTemp.index, columns=np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1 = expTemp.loc[:, genes1]
    temp2 = expTemp.loc[:, genes2]
    temp1.columns = np.arange(npairs)
    temp2.columns = np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = genePairs
    return (ans)


def pair_transform(expMat):
    pTab = makePairTab(expMat)
    npairs = len(pTab.index)
    ans = pd.DataFrame(0, index=expMat.index, columns=np.arange(npairs))
    genes1 = pTab['genes1'].values
    genes2 = pTab['genes2'].values
    expTemp = expMat.loc[:, np.unique(np.concatenate([genes1, genes2]))]
    ans = pd.DataFrame(0, index=expTemp.index, columns=np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1 = expTemp.loc[:, genes1]
    temp2 = expTemp.loc[:, genes2]
    temp1.columns = np.arange(npairs)
    temp2.columns = np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = genePairs
    return (ans)


def gnrBP(expDat, cellLabels, topX=50):
    myPatternG = sc_sampR_to_pattern(cellLabels)
    levels = list(myPatternG.keys())
    ans = {}
    for i in range(0, len(levels)):
        xres = sc_testPattern(myPatternG[levels[i]], expDat)
        tmpAns = findBestPairs(xres, topX)
        ans[levels[i]] = tmpAns
    return ans


def ptGetTop(expDat, cell_labels, cgenes_list=None, topX=50, sliceSize=5000, quickPairs=True):
    if not quickPairs:
        genes = expDat.columns.values
        grps = np.unique(cell_labels)
        myPatternG = sc_sampR_to_pattern(cell_labels)
        pairTab = makePairTab(genes)
        nPairs = len(pairTab)
        start = 0
        stp = np.min([sliceSize, nPairs])
        tmpTab = pairTab.iloc[start:stp, :]
        tmpPdat = ptSmall(expDat, tmpTab)
        statList = {k: sc_testPattern(v, tmpPdat) for k, v in myPatternG.items()}
        start = stp
        stp = start + sliceSize
        while start < nPairs:
            print(start)
            if stp > nPairs:
                stp = nPairs
            tmpTab = pairTab.iloc[start:stp, :]
            tmpPdat = ptSmall(expDat, tmpTab)
            tmpAns = {k: sc_testPattern(v, tmpPdat) for k, v in myPatternG.items()}
            for g in grps:
                statList[g] = pd.concat([statList[g], tmpAns[g]])
            start = stp
            stp = start + sliceSize
        res = []
        for g in grps:
            tmpAns = findBestPairs(statList[g], topX)
            res.append(tmpAns)
        return np.unique(np.array(res).flatten())

    else:
        myPatternG = sc_sampR_to_pattern(cell_labels)
        res = []
        grps = np.unique(cell_labels)
        for g in grps:
            print(g)
            genes = cgenes_list[g]
            pairTab = makePairTab(genes)
            nPairs = len(pairTab)
            tmpPdat = ptSmall(expDat, pairTab)
            tmpAns = findBestPairs(sc_testPattern(myPatternG[g], tmpPdat), topX)
            res.append(tmpAns)
        return np.unique(np.array(res).flatten())


def findClassyGenes(expDat, sampTab, dLevel, topX=25, dThresh=0, alpha1=0.05, alpha2=.001, mu=2):
    gsTrain = sc_statTab(expDat, dThresh=dThresh)
    ggenes = sc_filterGenes(gsTrain, alpha1=alpha1, alpha2=alpha2, mu=mu)
    grps = sampTab[dLevel]
    xdiff = gnrAll(expDat.loc[:, ggenes], grps)
    groups = np.unique(grps)
    res = []
    cgenes = {}
    for g in groups:
        temp = getClassGenes(xdiff[g], topX)
        cgenes[g] = temp
        res.append(temp)
    cgenes2 = np.unique(np.array(res).flatten())
    return [cgenes2, grps, cgenes]


#######################################################
#For Cell Type Deconvolution
#######################################################


def rowNormalizeFeatures(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for scGCN model and conversion to tuple
    representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized  #sparse_to_tuple(adj_normalized)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_pseudo_real_data(counts, log=True):
    for i in range(len(counts)):
        #norm_counts = sc.pp.normalize_total(counts[i])
        normalize(adata=counts[i], counts_per_cell_after=1e4)
        if log:
            sc.pp.log1p(counts[i])


def get_pVal(counts, labels):
    if isinstance(counts, sp.csr_matrix):
        y = counts.toarray().flatten()
    else:
        y = counts.flatten()
    sub_g = pd.DataFrame({'y': y, 'x': labels})
    lm = ols('y ~ x', data=sub_g).fit()
    pval = sm.stats.anova_lm(lm, typ=1).loc[['x'], 'PR(>F)'][0]
    #print(pval)
    return (pval)


#' select variable genes
def select_var_genes_anova(counts, labels, clust_vr='cellType', nv=2000):

    # D: number of genes
    D = counts.shape[1]

    #fit cell labels to cell expression, indidually for each gene
    new_labels = labels[clust_vr]
    #for each gene (column), anova fit cell labels to the gene's expression across cells (rows)
    #get p-value corrected bonferroni for each gene
    pv1 = [get_pVal(counts[:, d], new_labels) for d in range(D)]

    #get indices of nv genes with highest pVal
    egen = sorted(range(len(pv1)), key=lambda i: pv1[i], reverse=True)[:nv]

    return (egen)


def gen_mix(sc_counts, nc_min=2, nc_max=10, clust_vr='cellType', umi_cutoff=25000, downsample_counts=20000):
    all_subclasses = sc_counts.obs[clust_vr].unique().tolist().copy()
    mix_counts = sc_counts.copy()

    # sample between 2 and 10 cells randomly from the sc count matrix
    #sc.pp.subsample(mix_counts, n_obs=n_mix, random_seed=None)
    n_mix = random.choice(range(nc_min, nc_max + 1))
    sample_inds = np.random.choice(10, size=n_mix, replace=False)
    mix_counts = mix_counts[sorted(sample_inds)]

    # Combine (sum) their transcriptomic info (counts)
    #downsample > 25k counts to <= 20k counts
    #if np.sum(mix_counts.X) > umi_cutoff:
    #    sc.pp.downsample_counts(mix_counts, total_counts=downsample_counts)
    if isinstance(mix_counts.X, sp.csr_matrix):
        mix_counts_X = sp.csr_matrix.sum(mix_counts.X, axis=0)
    else:
        mix_counts_X = np.sum(mix_counts.X, axis=0, keepdims=True)

    subclasses = mix_counts.obs[clust_vr].unique().tolist().copy()

    class_prop = mix_counts.obs[clust_vr].value_counts(normalize=True).sort_index().to_frame().T.reset_index(drop=True)
    obs = class_prop.copy()
    for subcl in list(set(all_subclasses) - set(class_prop.columns.tolist())):
        obs[subcl] = 0.0
    obs = obs.sort_index(axis=1)
    obs['cell_count'] = n_mix
    obs['total_umi_count'] = np.sum(mix_counts.X)
    return (mix_counts_X, obs)


def gen_pseudo_spots(sc_counts, labels, clust_vr='cellType', nc_min=2, nc_max=10, N_p=1000, seed=0):
    np.random.seed(seed)
    tmp_sc_cnt = sc_counts.copy()

    mix_X = np.empty((0, tmp_sc_cnt.n_vars))
    mix_obs = pd.DataFrame()
    for i in range(N_p):
        #gets and combines a random mix of nc_min to nc_max cells
        mix_counts, obs = gen_mix(tmp_sc_cnt, clust_vr=clust_vr, umi_cutoff=25000, downsample_counts=20000)
        #append this mix to sample of pseudo mixtures
        mix_X = np.append(mix_X, mix_counts, axis=0)
        mix_obs = mix_obs.append(obs)

    mix_obs.index = pd.Index(['ps_mix_' + str(i + 1) for i in range(N_p)])
    #create AnnData object with sample of pseudo mixtures (obs)
    #annotations: cell counts, cell type compositions
    pseudo_counts = anndata.AnnData(X=mix_X, obs=mix_obs, var=sc_counts.var, dtype=mix_X.dtype)
    return (pseudo_counts)


#' This function takes pseudo-spatail and real-spatial data to identify variable genes
def pseudo_spatial_process(counts, labels, clust_vr='cellType', scRNA=False, n_hvg=2000, N_p=500):
    #labels: sample (mix or spot) cell compositions
    #counts: anndata type - one for scRNA or pseudo data, and one for real ST data
    st_counts = [counts[0].copy(), counts[1].copy()]
    #use common genes only
    genes1 = set(st_counts[0].var.index)
    genes2 = set(st_counts[1].var.index)
    intersect_genes = genes1.intersection(genes2)
    st_counts[0] = st_counts[0][:, list(intersect_genes)]
    st_counts[1] = st_counts[1][:, list(intersect_genes)]

    #if using scRNA data, generate pseudo ST data
    if (scRNA):
        print('generate pseudo ST from scRNA')
        #get top nv variable genes in scRNA set
        #sel_features = select_var_genes_anova(counts[0].X,labels[0],clust_vr=clust_vr, nv=n_hvg)
        logX = st_counts[0].copy()
        sc.pp.log1p(logX)
        sc.pp.highly_variable_genes(logX, flavor='seurat', n_top_genes=n_hvg)
        sel_features = list(logX.var[logX.var.highly_variable == True].index)

        #subset on top nv genes
        st_counts = [st_counts[0][:, sel_features], st_counts[1][:, sel_features]]

        #generate pseudo spots from scRNA (using variable genes) - as AnnData object
        st_counts = [gen_pseudo_spots(st_counts[0], labels[0], clust_vr=clust_vr, N_p=N_p), st_counts[1]]

    #otherwise, already using pseudo ST data -->

    #library size normalization of the pseudo and real ST data
    #log indicates to apply log1p transformation after normalization
    normalize_pseudo_real_data(st_counts, log=True)

    #find highly variable genes for both pseudo and real data
    #flavors: seurat_v3 - expects raw count data
    #flavors: seurat (default), cell_ranger - expect log data
    batch_cnt = st_counts[0].concatenate(st_counts[1], index_unique=None)

    sc.pp.highly_variable_genes(batch_cnt, flavor='seurat', n_top_genes=n_hvg, batch_key='batch')

    hvgs = list(batch_cnt.var[batch_cnt.var.highly_variable == True].index)

    st_counts[0]._inplace_subset_var(np.array(batch_cnt.var.highly_variable == True))
    st_counts[1]._inplace_subset_var(np.array(batch_cnt.var.highly_variable == True))

    #scale/standardize pseudo and real ST data
    sc.pp.scale(st_counts[0])
    sc.pp.scale(st_counts[1])

    labels1 = st_counts[0].obs  #.drop(['cell_count', 'total_umi_count'], axis=1)
    labels2 = pd.DataFrame(np.resize(labels1.values, (st_counts[1].shape[0], labels1.shape[1])),
                           columns=labels1.columns, index=st_counts[1].obs.index)
    labels = [labels1.reset_index(drop=True), labels2.reset_index(drop=True)]

    print('Finished  pre-processing')
    return ([st_counts[0], st_counts[1]], labels, hvgs)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def split(st_data, lab_label, pre_process, split_val=.9):

    data1 = pd.DataFrame(st_data[0].X, index=st_data[0].obs.index, columns=st_data[0].var.index)
    data2 = pd.DataFrame(st_data[1].X, index=st_data[1].obs.index, columns=st_data[1].var.index)

    lab_label1 = lab_label[0]
    lab_label2 = lab_label[1]

    lab_data1 = data1.reset_index(drop=True)  #.transpose()
    lab_data2 = data2.reset_index(drop=True)  #.transpose()

    random.seed(123)
    p_data = lab_data1
    p_label = lab_label1

    temd_train, temD_val, teml_train, temL_val = train_test_split(p_data, p_label, test_size=0.2, random_state=1)
    temd_val, temd_test, teml_val, teml_test = train_test_split(temD_val, temL_val, test_size=0.5, random_state=1)

    print((temd_train.index == teml_train.index).all())
    print((temd_test.index == teml_test.index).all())
    print((temd_val.index == teml_val.index).all())
    data_train = temd_train
    label_train = teml_train
    data_test = temd_test
    label_test = teml_test
    data_val = temd_val
    label_val = teml_val

    data_train1 = data_train
    data_test1 = data_test
    data_val1 = data_val
    label_train1 = label_train
    label_test1 = label_test
    label_val1 = label_val

    train2 = pd.concat([data_train1, lab_data2])
    lab_train2 = pd.concat([label_train1, lab_label2])

    datas_train = np.array(train2)
    datas_test = np.array(data_test1)
    datas_val = np.array(data_val1)
    labels_train = np.array(lab_train2)
    labels_test = np.array(label_test1)
    labels_val = np.array(label_val1)

    #' convert pandas data frame to csr_matrix format
    datas_tr = scipy.sparse.csr_matrix(datas_train.astype(np.float64))
    datas_va = scipy.sparse.csr_matrix(datas_val.astype(np.float64))
    datas_te = scipy.sparse.csr_matrix(datas_test.astype(np.float64))

    M = len(data_train1)

    #' 4) get the feature object by combining training, test, valiation sets
    features = sp.vstack((sp.vstack((datas_tr, datas_va)), datas_te)).tolil()
    if pre_process:
        features = rowNormalizeFeatures(features)

    labels_tr = labels_train
    labels_va = labels_val
    labels_te = labels_test

    labels = np.concatenate([np.concatenate([labels_tr, labels_va]), labels_te])
    Labels = pd.DataFrame(labels)

    true_label = Labels

    #' new label with binary values
    new_label = labels
    idx_train = range(M)
    idx_pred = range(M, len(labels_tr))
    idx_val = range(len(labels_tr), len(labels_tr) + len(labels_va))
    idx_test = range(len(labels_tr) + len(labels_va), len(labels_tr) + len(labels_va) + len(labels_te))

    train_mask = sample_mask(idx_train, new_label.shape[0])
    pred_mask = sample_mask(idx_pred, new_label.shape[0])
    val_mask = sample_mask(idx_val, new_label.shape[0])
    test_mask = sample_mask(idx_test, new_label.shape[0])

    labels_binary_train = np.zeros(new_label.shape)
    labels_binary_val = np.zeros(new_label.shape)
    labels_binary_test = np.zeros(new_label.shape)
    labels_binary_train[train_mask, :] = new_label[train_mask, :]
    labels_binary_val[val_mask, :] = new_label[val_mask, :]
    labels_binary_test[test_mask, :] = new_label[test_mask, :]

    adj_data = [data_train1, data_val1, data_test1, labels, lab_data2]

    return adj_data, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label


def l2norm(mat):
    stat = np.sqrt(np.sum(mat**2, axis=1))
    cols = mat.columns
    mat[cols] = mat[cols].div(stat, axis=0)
    mat[np.isinf(mat)] = 0
    return mat


#' @param num.cc Number of canonical vectors to calculate
#' @param seed.use Random seed to set.
#' @importFrom SVD
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


def filter_data(X, highly_genes=500):
    """Remove less variable genes.

    Parameters
    ----------
    X :
        cell-gene data.
    highly_genes : int optional
        number of chosen genes.

    Returns
    -------
    genes_idx :
        index of chosen genes
    cells_idx :
        index of chosen cells

    """

    X = np.ceil(X).astype(int)
    adata = sc.AnnData(X, dtype=np.float32)

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, flavor='cell_ranger', min_disp=0.5,
                                n_top_genes=highly_genes, subset=True)
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)

    return genes_idx, cells_idx


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


def geneSelection(data, threshold=0, atleast=10, yoffset=.02, xoffset=5, decay=1.5, n=None, plot=True, markers=None,
                  genes=None, figsize=(6, 3.5), markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    if sparse.issparse(data):
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

    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + .1, .1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected), xoffset, yoffset),
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(
                .4, 0.2,
                '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected), decay, xoffset,
                                                                               yoffset), color='k', fontsize=labelsize,
                transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:, None], y[:, None]), axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i] + dx + .1, zeroRate[i] + dy, g, color='k', fontsize=labelsize)

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


def normalize_adata(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


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


#######################################################
#For Spatial Domain
#######################################################

import math

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class ImageModel:

    def __init__(self, name='resnet50', device='cpu'):
        support_model_name = {"resnet50", "inception_v3", "xception", "vgg16"}
        if name not in support_model_name:
            ValueError("{} is not supported".format(name))
        self.model = getattr(tv.models, name)(pretrained=True)
        self.model.fc = torch.nn.Sequential()
        self.model = self.model.to(device)
        self.mean = np.array([0.406, 0.485, 0.456])
        self.std = np.array([0.225, 0.229, 0.224])

    def preprocess(self, x):
        # x: cv2 image
        # y = (x - mean) / std
        x = (x - self.mean) / self.std
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        return x

    def encoder(self, x):
        # extract last linear layers
        feature = self.model(x)
        # flatten
        feature = feature.view(feature.size(0), -1)
        return feature


def extract_feature(adata, image, cnn_base="resnet50", n_components=50, verbose=False, seeds=1, crop_size=20,
                    cnn_target_size=299, device="cpu", test=False):
    if test:
        fake_feature = np.random.rand(adata.n_obs, n_components)
        fake_pca_img_feature = np.random.rand(adata.n_obs, n_components)
        adata.obsm["X_tile_feature"] = fake_feature
        adata.obsm["X_morphology"] = fake_pca_img_feature
        return

    feature_df = pd.DataFrame()
    model = ImageModel(cnn_base)
    model.model = model.model.to(device)
    for i in tqdm(range(len(adata)), desc="Extracting feature", bar_format="{l_bar}{bar} [ time left: {remaining} ]"):

        x = adata.obs["x_pixel"][i]
        y = adata.obs["y_pixel"][i]

        img = image[x - crop_size:x + crop_size, y - crop_size:y + crop_size, :]
        img = cv2.resize(img, (cnn_target_size, cnn_target_size))
        img = model.preprocess(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        feature = model.encoder(img)
        feature = feature.detach().cpu().numpy()
        feature_df[i] = feature.reshape(-1)

    tile_feature = feature_df.transpose().to_numpy()

    pca = PCA(n_components=n_components, random_state=seeds)
    pca_img_feature = pca.fit_transform(tile_feature)
    adata.obsm["X_tile_feature"] = tile_feature
    adata.obsm["X_morphology"] = pca_img_feature


def calculate_weight_matrix(
    adata,
    adata_imputed=None,
    pseudo_spots=False,
    platform="Visium",
):
    # change to our dataframe
    # get pixel and coordinate
    img_row = adata.obs["imagerow"]
    img_col = adata.obs["imagecol"]
    array_row = adata.obs["array_row"]
    array_col = adata.obs["array_col"]
    rate = 3

    reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)

    reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)

    if pseudo_spots and adata_imputed:
        pd = pairwise_distances(
            adata_imputed.obs[["imagecol", "imagerow"]],
            adata.obs[["imagecol", "imagerow"]],
            metric="euclidean",
        )
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
        pd_norm = np.where(pd >= unit, 0, 1)

        md = 1 - pairwise_distances(
            adata_imputed.obsm["X_morphology"],
            adata.obsm["X_morphology"],
            metric="cosine",
        )
        md[md < 0] = 0

        adata_imputed.uns["physical_distance"] = pd_norm
        adata_imputed.uns["morphological_distance"] = md

        adata_imputed.uns["weights_matrix_all"] = (adata_imputed.uns["physical_distance"] *
                                                   adata_imputed.uns["morphological_distance"])

    else:
        pd = pairwise_distances(adata.obs[["imagecol", "imagerow"]], metric="euclidean")
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
        pd_norm = np.where(pd >= rate * unit, 0, 1)

        md = 1 - pairwise_distances(adata.obsm["X_morphology"], metric="cosine")
        md[md < 0] = 0

        gd = 1 - pairwise_distances(adata.obsm["X_pca"], metric="correlation")
        adata.uns["gene_expression_correlation"] = gd
        adata.uns["physical_distance"] = pd_norm
        adata.uns["morphological_distance"] = md

        adata.uns["weights_matrix_all"] = (adata.uns["physical_distance"] * adata.uns["morphological_distance"] *
                                           adata.uns["gene_expression_correlation"])
        adata.uns["weights_matrix_pd_gd"] = (adata.uns["physical_distance"] * adata.uns["gene_expression_correlation"])
        adata.uns["weights_matrix_pd_md"] = (adata.uns["physical_distance"] * adata.uns["morphological_distance"])
        adata.uns["weights_matrix_gd_md"] = (adata.uns["gene_expression_correlation"] *
                                             adata.uns["morphological_distance"])


def pca(adata, n_components=50, seeds=1):
    pca = PCA(n_components=n_components, random_state=seeds)
    pca.fit(adata.X.toarray())
    adata.obsm["X_pca"] = pca.transform(adata.X.toarray())


def impute_neighbour(
    adata: AnnData,
    count_embed=None,
    weights="weights_matrix_all",
    copy=False,
):
    coor = adata.obs[["imagecol", "imagerow"]]

    weights_matrix = adata.uns[weights]

    lag_coor = []

    weights_list = []

    with tqdm(
            total=len(adata),
            desc="Adjusting data",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(len(coor)):

            main_weights = weights_matrix[i]

            if weights == "physical_distance":
                current_neighbour = main_weights.argsort()[-6:]
            else:
                current_neighbour = main_weights.argsort()[-3:]

            surrounding_count = count_embed[current_neighbour]
            surrounding_weights = main_weights[current_neighbour]
            if surrounding_weights.sum() > 0:
                surrounding_weights_scaled = (surrounding_weights / surrounding_weights.sum())
                weights_list.append(surrounding_weights_scaled)

                surrounding_count_adjusted = np.multiply(surrounding_weights_scaled.reshape(-1, 1), surrounding_count)
                surrounding_count_final = np.sum(surrounding_count_adjusted, axis=0)

            else:
                surrounding_count_final = np.zeros(count_embed.shape[1])
                weights_list.append(np.zeros(len(current_neighbour)))
            lag_coor.append(surrounding_count_final)
            pbar.update(1)

    imputed_data = np.array(lag_coor)
    key_added = "imputed_data"
    adata.obsm[key_added] = imputed_data

    adata.obsm["top_weights"] = np.array(weights_list)

    return adata if copy else None


def SME_normalize(
    adata,
    use_data="raw",
    weights="weights_matrix_all",
    platform="Visium",
    copy=False,
):
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            count_embed = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            count_embed = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            count_embed = adata.X.values
        else:
            raise ValueError(f"""\
                    {type(adata.X)} is not a valid type.
                    """)
    else:
        count_embed = adata.obsm[use_data]

    calculate_weight_matrix(adata, platform=platform)

    impute_neighbour(adata, count_embed=count_embed, weights=weights)

    imputed_data = adata.obsm["imputed_data"].astype(float)
    imputed_data[imputed_data == 0] = np.nan
    adjusted_count_matrix = np.nanmean(np.array([count_embed, imputed_data]), axis=0)

    key_added = use_data + "_SME_normalized"
    adata.obsm[key_added] = adjusted_count_matrix

    print("The data adjusted by SME is added to adata.obsm['" + key_added + "']")

    return adata if copy else None


# define scale function to replace scanpy.pp.scale
# scale function is used to normalize the data by the mean and standard deviation of the data and clip the data to the range of max and min
def scale(adata, maxvalue=None, use_raw=False, copy=False):
    # adata.X is sparse
    try:
        data = adata.X.toarray()
    except:
        data = adata.X
    mean, std = data.mean(axis=0), data.std(axis=0)
    adata.X = (data - mean) / (std + 0.00001)
    if maxvalue is not None:
        adata.X = np.clip(adata.X, -maxvalue, maxvalue)
    if use_raw:
        adata.raw = adata.copy()
    return adata if copy else None


def sc_scale(adata, maxvalue=None, use_raw=False, copy=False):
    # adata.X is sparse
    sc.pp.scale(adata)


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

    if model == 'GraphSCI':
        ### Normalization ###
        sc.pp.filter_genes(adata, min_counts=params.min_counts)
        sc.pp.filter_cells(adata, min_counts=params.min_counts)
        sc.pp.highly_variable_genes(adata, n_top_genes=model_params.n_genes, flavor='seurat_v3', subset=True)
        print(str(adata.shape[0]) + " cells and " + str(adata.shape[1]) + " genes left.")

        sc.pp.log1p(adata)
        sc.pp.normalize_per_cell(adata)
        adata.raw = adata.copy()
        adata.var['size_factors'] = adata.var.n_counts / np.median(adata.var.n_counts)
        size_factors = adata.var.size_factors
        sc.pp.scale(adata)
        #####################
        num_cells = adata.shape[0]
        num_genes = adata.shape[1]

        ### train/test split ###
        train_idx, test_idx = train_test_split(np.arange(num_genes), test_size=model_params.test_size,
                                               random_state=random_seed)
        tt_label = pd.Series(['train'] * num_genes)
        tt_label.iloc[test_idx] = 'test'
        adata.var['dca_split'] = tt_label.values
        adata.var['dca_split'] = adata.var['dca_split'].astype('category')
        print('split')
        # train_data = adata.copy().X
        train_data = adata[:, adata.var['dca_split'] == 'train'].copy().X
        train_data_raw = adata.raw[:, adata.var['dca_split'] == 'train'].copy().X
        if sp.issparse(train_data_raw):
            train_data_raw = train_data_raw.toarray()
        test_data = adata.copy().X
        test_data_raw = adata.raw.copy().X
        if sp.issparse(test_data_raw):
            test_data_raw = test_data_raw.toarray()
        train_size_factors = adata[:, adata.var['dca_split'] == 'train'].var.size_factors
        test_size_factors = adata.var.size_factors

        ### create graph ###
        g2g_corr = np.corrcoef(train_data, rowvar=False)
        g2g_corr[np.isnan(g2g_corr)] = 1  # replace nan with 1 corr for 0 variance columns with all same entries
        adj_matrix = np.zeros([train_data.shape[1], train_data.shape[1]])
        adj_matrix[np.abs(g2g_corr) > model_params.gene_corr] = 1
        adj_train = adj_matrix

        g2g_corr_test = np.corrcoef(test_data, rowvar=False)
        g2g_corr_test[np.isnan(
            g2g_corr_test)] = 1  # replace nan with 1 corr for 0 variance columns with all same entries
        adj_test = np.zeros([test_data.shape[1], test_data.shape[1]])
        adj_test[np.abs(g2g_corr_test) > model_params.gene_corr] = 1

        ## Create fake_test edges
        edges = np.where(adj_train)
        num_edges = len(edges[0])
        adj_train_false = np.zeros([adj_train.shape[0], adj_train.shape[1]])
        edges_dic = [(edges[0][i], edges[1][i]) for i in range(num_edges)]
        false_edges_dic = []
        """
        from itertools import product
        all_pairs = {i for i in product(range(adj_test.shape[0]), repeat=2)}
        false_edges = all_pairs.difference(edges_dic)
        false_edges = np.random.shuffle(np.array(list(false_edges)))[:num_edges]
        for i in range(num_edges):
            adj_test_false[false_edges[i, 0], false_edges[i, 1]] = 1
            adj_test_false[false_edges[i, 1], false_edges[i, 0]] = 1

        while adj_train_false.sum() < num_edges:
            i = np.random.randint(0, adj_train.shape[0])
            j = np.random.randint(0, adj_train.shape[0])
            if (i, j) in edges_dic:
                continue
            if (j, i) in edges_dic:
                continue
            if (i, j) in false_edges_dic:
                continue
            if (j, i) in false_edges_dic:
                continue
            else:
                false_edges_dic.append((i, j))
                false_edges_dic.append((j, i))
            # if np.random.random_sample() > 0.333: # unnecessary but in original
                adj_train_false[i, j] = 1
                # adj_test_false[j, i] = 1 #Not in original code ..? why

        while adj_test_false.sum() < num_edges:
            i = np.random.randint(0, adj_test.shape[0])
            j = np.random.randint(0, adj_test.shape[0])
            false_edges_dic.append((i, j))
            false_edges_dic.append((j, i))
            adj_test_false[i, j] = 1
            adj_test_false[j, i] = 1
        """
        vals = np.power(np.sum(adj_train, axis=1), -1 / 2)
        vals = np.expand_dims(vals, axis=0)
        adj_norm_train = vals * adj_train
        adj_norm_train = vals.T * adj_norm_train
        vals = np.power(np.sum(adj_test, axis=1), -1 / 2)
        vals = np.expand_dims(vals, axis=0)
        adj_norm_test = vals * adj_test
        adj_norm_test = vals.T * adj_norm_test
        device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')
        train_data = torch.Tensor(train_data.T).to(device)
        train_data_raw = torch.Tensor(train_data_raw.T).to(device)
        test_data = torch.Tensor(test_data.T).to(device)
        test_data_raw = torch.Tensor(test_data_raw.T).to(device)
        adj_test = torch.Tensor(adj_test).to(device)
        adj_train = torch.Tensor(adj_train).to(device)
        adj_norm_train = torch.Tensor(adj_norm_train).to(device)
        adj_norm_test = torch.Tensor(adj_norm_test).to(device)
        train_size_factors = torch.Tensor(train_size_factors).to(device)
        test_size_factors = torch.Tensor(test_size_factors).to(device)

        data_dict = {}
        data_dict['num_cells'] = num_cells
        data_dict['num_genes'] = num_genes
        data_dict['adj_train'] = adj_train
        data_dict['adj_norm_train'] = adj_norm_train
        data_dict['adj_norm_test'] = adj_norm_test
        data_dict['adj_test'] = adj_test
        data_dict['adj_train_false'] = adj_train_false
        data_dict['train_data'] = train_data
        data_dict['train_data_raw'] = train_data_raw
        data_dict['test_data'] = test_data
        data_dict['test_data_raw'] = test_data_raw
        data_dict['adata'] = adata
        data_dict['size_factors'] = size_factors
        data_dict['train_size_factors'] = train_size_factors
        data_dict['test_size_factors'] = test_size_factors
        data_dict['test_idx'] = test_idx
        ###########################

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

    if model == 'scGNN':

        num_cells = adata.shape[0]
        num_genes = adata.shape[1]

        min_genes = .01 * num_genes  # make parameter
        min_cells = .01 * num_cells  # make parameter
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.log1p(adata)
        adata.raw = adata.copy()
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
        adata = adata[:, adata.var['highly_variable']]

        ### train/test split ###
        genelist = adata.var_names
        celllist = adata.obs_names
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=random_seed)
        tt_label = pd.Series(['train'] * adata.n_obs)
        tt_label.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = tt_label.values
        adata.obs['dca_split'] = adata.obs['dca_split'].astype('category')

        train_data = adata[adata.obs['dca_split'] == 'train', :].copy().X
        test_data = adata.copy().X
        if sp.issparse(test_data):
            test_data = test_data.toarray()

        data_dict = {}
        data_dict['num_cells'] = num_cells
        data_dict['num_genes'] = num_genes
        data_dict['train_data'] = train_data
        data_dict['test_data'] = test_data
        data_dict['adata'] = adata
        data_dict['genelist'] = genelist
        data_dict['celllist'] = celllist
        data_dict['test_idx'] = test_idx

    return data_dict


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.
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

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')

    # coor = pd.DataFrame(adata.obsm['spatial'])
    x_pixel = pd.DataFrame(adata.obs['x_pixel'])
    y_pixel = pd.DataFrame(adata.obs['y_pixel'])
    coor = pd.concat([x_pixel, y_pixel], axis=1)
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
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0, ]
    id_cell_trans = dict(zip(
        range(coor.shape[0]),
        np.array(coor.index),
    ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis, key_section='Section_id', section_order=None,
                       verbose=True):
    """\
    Construct the spatial neighbor networks.
    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.

    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating 2D SNN of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section, ]
        Cal_Spatial_Net(temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' % (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        adata.uns['Spatial_Net_2D'] = pd.concat([adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
    for it in range(num_section - 1):
        section_1 = section_order[it]
        section_2 = section_order[it + 1]
        if verbose:
            print('------Calculating SNN between adjacent section {} and {}.'.format(section_1, section_2))
        Z_Net_ID = section_1 + '-' + section_2
        temp_adata = adata[adata.obs[key_section].isin([section_1, section_2]), ]
        Cal_Spatial_Net(temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        spot_section_trans = dict(zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(spot_section_trans)
        used_edge = temp_adata.uns['Spatial_Net'].apply(lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge, ]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, ['Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' % (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        adata.uns['Spatial_Net_Zaxis'] = pd.concat([adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
    adata.uns['Spatial_Net'] = pd.concat([adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' % (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (adata.uns['Spatial_Net'].shape[0] / adata.n_obs))
