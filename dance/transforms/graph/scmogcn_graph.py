import os
import pickle
from collections import defaultdict

import dgl
import numpy as np
import scipy.sparse
import torch
from sklearn.decomposition import TruncatedSVD

from dance import logger
from dance.data.base import Data
from dance.typing import Union

from ..base import BaseTransform


def read_gmt(entrez_string, symbol_string):
    gene_list = entrez_string.split()

    gene_sets_entrez = defaultdict(list)
    indicator = 0
    for ele in gene_list:
        if ele.isnumeric():
            gene_sets_entrez[gene_set_name].append(ele)
        elif indicator == 1:
            indicator = 0
        elif indicator == 0:
            indicator = 1
            gene_set_name = ele

    gene_list = symbol_string.split()
    gene_sets_symbols = defaultdict(list)

    for ele in gene_list:
        if ele in gene_sets_entrez:
            gene_set_name = ele
        elif not ele.startswith('http://'):
            gene_sets_symbols[gene_set_name].append(ele)
    return gene_sets_symbols


def create_pathway_graph(gex_features: scipy.sparse.spmatrix, gene_names: Union[str], pathway_weight: str,
                         pathway_threshold: float, subtask: str, pathway_path: str):
    """Generate nodes, edges and edge weights for pathway dataset.

    Parameters
    ----------
    gex_features: scipy.sparse.spmatrix
        Gene expression data.
    gene_names: Union[str]
        Gene names.
    pathway_weight: str
        Weighting scheme for pathway filtering.
    pathway_threshold: float
        Threshold for pathway filtering.
    subtask: str
        Subtask name.
    pathway_path: str
        Path to pathway file.

    Returns
    --------
    uu: List[int]
        Predecessor node id of each edge.
    vv: List[int]
        Successor node id of each edge.
    ee: List[float]
        Edge weight of each edge.

    """

    uu = []
    vv = []
    ee = []

    pk_path = f'pw_{subtask}_{pathway_weight}.pkl'
    if os.path.exists(pk_path):
        logger.warning("Pathway file exist. Load pickle file by default. "
                       "Auguments '--pathway_weight' and '--pathway_path' will not take effect.")
        uu, vv, ee = pickle.load(open(pk_path, 'rb'))
    else:
        # Load Original Pathway File
        with open(pathway_path + '.entrez.gmt') as gmt:
            entrez_string = gmt.read()
        with open(pathway_path + '.symbols.gmt') as gmt:
            symbols_string = gmt.read()
        gene_sets_symbols = read_gmt(entrez_string, symbols_string)

        pw = [i[1] for i in gene_sets_symbols.items()]
        # Generate New Pathway Data
        new_pw = []
        for i in pw:
            new_pw.append([])
            for j in i:
                if j in gene_names:
                    new_pw[-1].append(gene_names.index(j))

        if pathway_weight == 'cos':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            sj = np.sqrt(np.dot(gex_features[:, j].T, gex_features[:, j]).item())
                            sk = np.sqrt(np.dot(gex_features[:, k].T, gex_features[:, k]).item())
                            jk = np.dot(gex_features[:, j].T, gex_features[:, k])
                            cossim = jk / sj / sk
                            ee.append(cossim.item())
        elif pathway_weight == 'one':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(1.)
        elif pathway_weight == 'pearson':
            corr = np.corrcoef(gex_features.T.todense())
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(corr[j][k])

        pickle.dump([uu, vv, ee], open(pk_path, 'wb'))

    # Apply Threshold
    nu = []
    nv = []
    ne = []

    for i in range(len(uu)):
        if abs(ee[i]) > pathway_threshold:
            ne.append(ee[i])
            nu.append(uu[i])
            nv.append(vv[i])
    uu, vv, ee = nu, nv, ne

    return uu, vv, ee


def construct_enhanced_feature_graph(u, v, e, train_size, feature_size, cell_node_features, inductive=False,
                                     enhance_graph=None, _test_graph=False):
    """Generate a feature-cell graph, enhanced with domain-knowledge (e.g. pathway).

    Parameters
    ----------
    u: torch.Tensor
        1-dimensional tensor. Cell node id of each cell-feature edge.
    v: torch.Tensor
        1-dimensional tensor. Feature node id of each cell-feature edge.
    e: torch.Tensor
        1-dimensional tensor. Weight of each cell-feature edge.
    train_size: int
        Number of training cells.
    feature_size: int
        Number of features.
    cell_node_features: torch.Tensor
        Cell node initial features.
    inductive: bool
        Whether to use inductive learning.
    enhance_graph: dgl.DGLGraph
        Enhanced graph.
    _test_graph: bool
        Whether to use test graph.


    Returns
    --------
    graph: DGLGraph
        The generated graph.

    """

    if enhance_graph is None:

        graph_data = {
            ('cell', 'cell2feature', 'feature'): (u, v),
            ('feature', 'feature2cell', 'cell'): (v, u),
        }

        graph = dgl.heterograph(graph_data)

        if inductive:
            graph.nodes['cell'].data['id'] = cell_node_features[:train_size] if not _test_graph else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        feature_size = min(graph.num_nodes('feature'), feature_size)
        graph.nodes['feature'].data['id'] = torch.arange(feature_size).long()
        graph.edges['feature2cell'].data['weight'] = e
        graph.edges['cell2feature'].data['weight'] = e[:graph.edges(etype='cell2feature')[0].shape[0]]

    else:
        uu, vv, ee = enhance_graph

        graph_data = {
            ('cell', 'cell2feature', 'feature'): (u, v),
            ('feature', 'feature2cell', 'cell'): (v, u),
            ('feature', 'pathway', 'feature'): (uu, vv),
        }
        graph = dgl.heterograph(graph_data)

        if inductive:
            graph.nodes['cell'].data['id'] = cell_node_features[:train_size] if not _test_graph else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        graph.nodes['feature'].data['id'] = torch.arange(feature_size).long()
        graph.edges['feature2cell'].data['weight'] = e.float()
        graph.edges['cell2feature'].data['weight'] = e[:graph.edges(etype='cell2feature')[0].shape[0]].float()
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    return graph


class ScMoGNNGraph(BaseTransform):
    """Construct the cell-feature graph object for scmognn.

    Parameters
    ----------
    inductive: bool
        Whether to use inductive learning. Default: False.
    cell_init: str
        Initialization method for cell features. Default: 'none'.
    pathway: bool
        Whether to use pathway information. Default: True.
    subtask: str
        Subtask name. Default: 'gex2adt'.
    pathway_weight: str
        Weighting scheme for pathway filtering. Default: None.
    pathway_threshold: float
        Threshold for pathway filtering. Default: 0.
    pathway_path: str
        Path to pathway file. Default: 'data/h.all.v7.4'.


    Returns
    --------
    g: DGLGraph
        The generated graph.

    """

    def __init__(self, inductive: bool = False, cell_init: str = 'none', pathway=True,
                 subtask='openproblems_bmmc_cite_phase2_rna', pathway_weight=None, pathway_threshold: float = 0.,
                 pathway_path: str = 'data/h.all.v7.4', **kwargs):
        super().__init__(**kwargs)
        self.inductive = inductive
        self.cell_init = cell_init
        self.pathway = pathway
        self.subtask = subtask
        self.pathway_weight = pathway_weight
        self.pathway_threshold = pathway_threshold
        self.pathway_path = pathway_path

    def __call__(self, data: Data) -> Data:

        x_train, _ = data.get_train_data(return_type="numpy")
        x_train_sparse, _ = data.get_train_data(return_type="sparse")
        x_test_sparse, _ = data.get_test_data(return_type="sparse")

        cell_size = x_train_sparse.shape[0] + x_test_sparse.shape[0] if not self.inductive else x_train_sparse.shape[0]
        train_size = x_train_sparse.shape[0]
        feature_size = x_train_sparse.shape[1]

        if self.cell_init == 'none':
            cell_node_features = torch.ones(cell_size).long()
        elif self.cell_init == 'svd':
            embedder_mod1 = TruncatedSVD(n_components=100)
            X_train_np = embedder_mod1.fit_transform(x_train_sparse)
            X_test_np = embedder_mod1.transform(x_test_sparse)
            cell_node_features = torch.cat([torch.from_numpy(X_train_np), torch.from_numpy(X_test_np)], 0).float()
        if self.pathway:
            gene_names = data.data['mod1'].var_names.tolist()
            enhance_graph = create_pathway_graph(gex_features=x_train, gene_names=gene_names,
                                                 pathway_weight=self.pathway_weight,
                                                 pathway_threshold=self.pathway_threshold, subtask=self.subtask,
                                                 pathway_path=self.pathway_path)
        else:
            enhance_graph = None

        if self.inductive:
            u = torch.from_numpy(
                np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(x_train_sparse)], axis=0))
            v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in x_train_sparse], axis=0))
            e = torch.from_numpy(x_train_sparse.tocsr().data).float()
            g = construct_enhanced_feature_graph(u, v, e, train_size, feature_size, cell_node_features, self.inductive,
                                                 enhance_graph, _test_graph=False)

            u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(x_train_sparse)] + \
                                                [np.array(t.nonzero()[0] + i + train_size) for i, t in
                                                 enumerate(x_test_sparse)], axis=0))
            v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in x_train_sparse] + \
                                                [np.array(t.nonzero()[1]) for t in x_test_sparse], axis=0))
            e = torch.from_numpy(np.concatenate(
                [x_train_sparse.tocsr().data, x_test_sparse.tocsr().data], axis=0)).float()
            gtest = construct_enhanced_feature_graph(u, v, e, train_size, feature_size, cell_node_features,
                                                     self.inductive, enhance_graph, _test_graph=True)
            data.data.uns['g'] = g
            data.data.uns['gtest'] = gtest

        else:
            u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(x_train_sparse)] + \
                                                [np.array(t.nonzero()[0] + i + train_size) for i, t in
                                                 enumerate(x_test_sparse)], axis=0))
            v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in x_train_sparse] + \
                                                [np.array(t.nonzero()[1]) for t in x_test_sparse], axis=0))
            e = torch.from_numpy(np.concatenate(
                [x_train_sparse.tocsr().data, x_test_sparse.tocsr().data], axis=0)).float()
            g = construct_enhanced_feature_graph(u, v, e, train_size, feature_size, cell_node_features, self.inductive,
                                                 enhance_graph)
            data.data.uns['g'] = g

        return data
