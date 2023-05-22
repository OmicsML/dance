import os
import pickle
import time
from collections import defaultdict
from typing import List, Union

import anndata as ad
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import torch
from dgl import nn as dglnn
from scipy import sparse as sp
from scipy.sparse import csc_matrix
from scipy.spatial import distance, distance_matrix, minkowski_distance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import pairwise_distances as pair
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from torch.nn import functional as F

from dance import logger


def csr_cosine_similarity(input_csr_matrix):
    similarity = input_csr_matrix * input_csr_matrix.T
    square_mag = similarity.diagonal()
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    res = similarity.multiply(inv_mag).T.multiply(inv_mag)
    return res.toarray()


def cosine_similarity_gene(input_matrix):
    res = cosine_similarity(input_matrix)
    res = np.abs(res)
    return res


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    # beta to control the range of neighbourhood when calculate grey vale for one spot
    beta_half = round(beta / 2)
    g = []
    for i in range(len(x_pixel)):
        max_x = image.shape[0]
        max_y = image.shape[1]
        nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
                    max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
    return c3


def construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph=None, test=False, **kwargs):
    """Generate a feature-cell graph, enhanced with domain-knowledge (e.g. pathway).

    Parameters
    ----------
    u: torch.Tensor
        1-dimensional tensor. Cell node id of each cell-feature edge.
    v: torch.Tensor
        1-dimensional tensor. Feature node id of each cell-feature edge.
    e: torch.Tensor
        1-dimensional tensor. Weight of each cell-feature edge.
    cell_node_features: torch.Tensor
        1-dimensional or 2-dimensional tensor.  Node features for each cell node.
    enhance_graph: list[torch.Tensor]
        Node ids and edge weights of enhancement graph.

    Returns
    --------
    graph: DGLGraph
        The generated graph.

    """

    TRAIN_SIZE = kwargs['TRAIN_SIZE']
    FEATURE_SIZE = kwargs['FEATURE_SIZE']

    if enhance_graph is None:
        print('WARNING: Enhance graph disabled.')

    if kwargs['only_pathway'] and enhance_graph is not None:
        assert (kwargs['subtask'].find('rna') != -1)
        uu, vv, ee = enhance_graph

        graph_data = {
            ('feature', 'feature2cell', 'cell'): (v, u),
            ('feature', 'pathway', 'feature'): (uu, vv),
        }
        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features

        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['feature2cell'].data['weight'] = e
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    elif kwargs['no_pathway'] or kwargs['subtask'].find('rna') == -1 or enhance_graph is None:

        if kwargs['inductive'] == 'opt':
            print('Not supported.')
            # graph_data = {
            #     ('cell', 'cell2feature', 'feature'): (u, v) if not test else (
            #         u[:g.edges(etype='cell2feature')[0].shape[0]], v[:g.edges(etype='cell2feature')[0].shape[0]]),
            #     ('feature', 'feature2cell', 'cell'): (v, u),
            # }

        else:
            graph_data = {
                ('cell', 'cell2feature', 'feature'): (u, v),
                ('feature', 'feature2cell', 'cell'): (v, u),
            }

        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['feature2cell'].data['weight'] = e
        graph.edges['cell2feature'].data['weight'] = e[:graph.edges(etype='cell2feature')[0].shape[0]]

    else:
        assert (kwargs['subtask'].find('rna') != -1)
        uu, vv, ee = enhance_graph

        if kwargs['inductive'] == 'opt':
            print("Not supported.")
            # graph_data = {
            #     ('cell', 'cell2feature', 'feature'): (u, v) if not test else (
            #         u[:g.edges(etype='cell2feature')[0].shape[0]], v[:g.edges(etype='cell2feature')[0].shape[0]]),
            #     ('feature', 'feature2cell', 'cell'): (v, u),
            #     ('feature', 'pathway', 'feature'): (uu, vv),
            # }
        else:
            graph_data = {
                ('cell', 'cell2feature', 'feature'): (u, v),
                ('feature', 'feature2cell', 'cell'): (v, u),
                ('feature', 'pathway', 'feature'): (uu, vv),
            }
        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['feature2cell'].data['weight'] = e
        graph.edges['cell2feature'].data['weight'] = e[:graph.edges(etype='cell2feature')[0].shape[0]]
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    return graph


# TODO: haven't explained extra kwargs
def construct_pathway_graph(gex_data, **kwargs):
    """Generate nodes, edges and edge weights for pathway dataset.

    Parameters
    ----------
    gex_data: anndata.AnnData
        Gene data, contains feature matrix (.X) and feature names (.var['feature_types']).

    Returns
    --------
    uu: list[int]
        Predecessor node id of each edge.
    vv: list[int]
        Successor node id of each edge.
    ee: list[float]
        Edge weight of each edge.

    """

    pww = kwargs['pathway_weight']
    npw = kwargs['no_pathway']
    subtask = kwargs['subtask']
    pw_path = kwargs['pathway_path']
    uu = []
    vv = []
    ee = []

    assert (not npw)

    pk_path = f'pw_{subtask}_{pww}.pkl'
    #     pk_path = f'pw_{subtask}_{pww}.pkl'
    if os.path.exists(pk_path):
        print(
            'WARNING: Pathway file exist. Load pickle file by default. Auguments "--pathway_weight" and "--pathway_path" will not take effect.'
        )
        uu, vv, ee = pickle.load(open(pk_path, 'rb'))
    else:
        # Load Original Pathway File
        with open(pw_path + '.entrez.gmt') as gmt:
            gene_list = gmt.read().split()

        gene_sets_entrez = defaultdict(list)
        indicator = 0
        for ele in gene_list:
            if not ele.isnumeric() and indicator == 1:
                indicator = 0
                continue
            if not ele.isnumeric() and indicator == 0:
                indicator = 1
                gene_set_name = ele
            else:
                gene_sets_entrez[gene_set_name].append(ele)

        with open(pw_path + '.symbols.gmt') as gmt:
            gene_list = gmt.read().split()

        gene_sets_symbols = defaultdict(list)

        for ele in gene_list:
            if ele in gene_sets_entrez:
                gene_set_name = ele
            elif not ele.startswith('http://'):
                gene_sets_symbols[gene_set_name].append(ele)

        pw = [i[1] for i in gene_sets_symbols.items()]

        # Generate New Pathway Data
        counter = 0
        total = 0
        feature_index = gex_data.var['feature_types'].index.tolist()
        gex_features = gex_data.X
        new_pw = []
        for i in pw:
            new_pw.append([])
            for j in i:
                if j in feature_index:
                    new_pw[-1].append(feature_index.index(j))

        if pww == 'cos':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            sj = np.sqrt(np.dot(gex_features[:, j].toarray().T, gex_features[:, j].toarray()).item())
                            sk = np.sqrt(np.dot(gex_features[:, k].toarray().T, gex_features[:, k].toarray()).item())
                            jk = np.dot(gex_features[:, j].toarray().T, gex_features[:, k].toarray())
                            cossim = jk / sj / sk
                            ee.append(cossim.item())
        elif pww == 'one':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(1.)
        elif pww == 'pearson':
            corr = np.corrcoef(gex_features.toarray().T)
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(corr[j][k])

        pickle.dump([uu, vv, ee], open(pk_path, 'wb'))

    # Apply Threshold
    pwth = kwargs['pathway_threshold']
    nu = []
    nv = []
    ne = []

    for i in range(len(uu)):
        if abs(ee[i]) > pwth:
            ne.append(ee[i])
            nu.append(uu[i])
            nv.append(vv[i])
    uu, vv, ee = nu, nv, ne

    return uu, vv, ee


def construct_basic_feature_graph(feature_mod1, feature_mod1_test=None, bf_input=None, device='cuda'):
    input_train_mod1 = csc_matrix(feature_mod1)
    """Generate a feature-cell graph, enhanced with domain-knowledge (e.g. pathway).

    Parameters
    ----------
    feature_mod1 : torch.Tensor
        Features of input modality.
    feature_mod1_test : torch.Tensor optional
        Features of input modality of testing samples.
    bf_input : torch.Tensor optional
        Batch features, by default to be None. If undefined, cell nodes would not have initialization.
    device : str optional
        The device where the graph would be generated, by default to be 'cuda'.

    Returns
    --------
    g: DGLGraph
        The generated graph.

    """

    if feature_mod1_test is not None:
        input_test_mod1 = csc_matrix(feature_mod1_test)
        assert (input_test_mod1.shape[1] == input_train_mod1.shape[1])

        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + input_train_mod1.shape[0]) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        sample_size = input_train_mod1.shape[0] + input_test_mod1.shape[0]
        weights = torch.from_numpy(np.concatenate(
            [input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()

    else:
        u = torch.from_numpy(
            np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        sample_size = input_train_mod1.shape[0]
        weights = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data], axis=0)).float()

    graph_data = {
        ('cell', 'cell2feature', 'feature'): (u, v),
        ('feature', 'feature2cell', 'cell'): (v, u),
    }
    g = dgl.heterograph(graph_data)

    if bf_input:
        g.nodes['cell'].data['bf'] = gen_batch_features(bf_input)
    else:
        g.nodes['cell'].data['bf'] = torch.zeros(sample_size).float()

    g.nodes['cell'].data['id'] = torch.zeros(sample_size).long()  #torch.arange(sample_size).long()
    #     g.nodes['cell'].data['source'] =
    g.nodes['feature'].data['id'] = torch.arange(input_train_mod1.shape[1]).long()
    g.edges['cell2feature'].data['weight'] = g.edges['feature2cell'].data['weight'] = weights

    g = g.to(device)
    return g


def gen_batch_features(ad_inputs):
    """Generate statistical features for each batch in the input data, and assign batch
    features to each cell. This function returns batch features for each cell in all the
    input sub-datasets.

    Parameters
    ----------
    ad_inputs: list[anndata.AnnData]
        A list of AnnData object, each contains a sub-dataset.

    Returns
    --------
    batch_features: torch.Tensor
        A batch_features matrix, each row refers to one cell from the datasets. The matrix can be directly used as the
        node features of cell nodes.

    """

    cells = []
    columns = [
        'cell_mean', 'cell_std', 'nonzero_25%', 'nonzero_50%', 'nonzero_75%', 'nonzero_max', 'nonzero_count',
        'nonzero_mean', 'nonzero_std', 'batch'
    ]

    assert len(ad_inputs) < 10, "WARNING: Input of gen_bf_features should be a list of AnnData objects."

    for ad_input in ad_inputs:
        bcl = list(ad_input.obs['batch'])
        print(set(bcl))
        for i, cell in enumerate(ad_input.X):
            cell = cell.toarray()
            nz = cell[np.nonzero(cell)]
            if len(nz) == 0:
                print('Error: one cell contains all zero features.')
                exit()
            cells.append([
                cell.mean(),
                cell.std(),
                np.percentile(nz, 25),
                np.percentile(nz, 50),
                np.percentile(nz, 75),
                cell.max(),
                len(nz) / 1000,
                nz.mean(),
                nz.std(), bcl[i]
            ])

    cell_features = pd.DataFrame(cells, columns=columns)
    batch_source = cell_features.groupby('batch').mean().reset_index()
    batch_list = batch_source.batch.tolist()
    batch_source = batch_source.drop('batch', axis=1).to_numpy().tolist()
    b2i = dict(zip(batch_list, range(len(batch_list))))
    batch_features = []

    for ad_input in ad_inputs:
        for b in ad_input.obs['batch']:
            batch_features.append(batch_source[b2i[b]])

    batch_features = torch.tensor(batch_features).float()

    return batch_features


def generate_cell_features(
    data: Union[ad.AnnData, List[ad.AnnData]],
    *,
    group_batch: bool = False,
    batch_col_id: str = "batch",
) -> torch.Tensor:
    """Generate cell node features from anndata objects.

    Parameters
    ----------
    data: Union[anndata.AnnData, List[anndata.AnnData]]
        A list of or a single AnnData object(s).
    group_batch: bool
        If set to True, set features of cell within a batch to the mean values.
    batch_col_id: str
        Column ID corresponding to the batchs.

    Returns
    -------
    cell_features: torch.Tensor
        A cell feature matrix, each row represents the node features corresponding to a cell, generated based on
        the statistics of the cell's gene expression profiles.

    TODO
    ----
        Add option for providing call-backs for additional flexibility of generating different types of features.

    """
    data = data if isinstance(data, list) else [data]

    cells = []
    columns = [
        "cell_mean", "cell_std", "nonzero_25%", "nonzero_50%", "nonzero_75%", "nonzero_max", "nonzero_count",
        "nonzero_mean", "nonzero_std", "batch"
    ]

    for adata in data:
        bcl = adata.obs[batch_col_id].tolist()
        logger.info(f"Unique batches: {sorted(set(bcl))}")

        for i, cell in enumerate(adata.X):
            cell = cell.toarray()
            nz = cell[np.nonzero(cell)]

            if len(nz) == 0:
                logger.warning("Encountered a cell with all zero features.")
                cells.append([0] * (len(columns) - 1) + bcl[i])
            else:
                cells.append([
                    cell.mean(),
                    cell.std(),
                    np.percentile(nz, 25),
                    np.percentile(nz, 50),
                    np.percentile(nz, 75),
                    cell.max(),
                    len(nz) / 1000,
                    nz.mean(),
                    nz.std(),
                    bcl[i],
                ])

    features = pd.DataFrame(cells, columns=columns)
    logger.debug(f"features=\n{features}")

    if group_batch:
        batch_source = features.groupby("batch", as_index=False).mean()
        logger.info(f"Batch features:\n{batch_source.set_index('batch')}")

        # Assign cell features with corresponding reduced batch features
        b2i = {j: i for i, j in enumerate(batch_source["batch"].tolist())}
        batch_source = batch_source.drop("batch", axis=1).to_numpy()
        cell_batch_idxs = list(map(b2i.get, features["batch"]))
        cell_features = batch_source[cell_batch_idxs]

    else:
        cell_features = features.drop("batch", axis=1).to_numpy()

    return torch.tensor(cell_features, dtype=torch.float32)


def construct_modality_prediction_graph(dataset, **kwargs):
    """Construct the cell-feature graph object for modality prediction task, based on
    the input dataset.

    Parameters
    ----------
    dataset: datasets.multimodality.ModalityPredictionDataset
        The input dataset, typically includes four input AnnData sub-datasets, which are train_mod1, train_mod2,
        test_mod1 and test_mod2 respectively.

    Returns
    --------
    g: DGLGraph
        The generated graph.

    """

    train_mod1 = dataset.modalities[0]
    input_train_mod1 = dataset.sparse_features()[0]
    if kwargs['inductive'] == 'trans':
        input_test_mod1 = dataset.sparse_features()[2]

    CELL_SIZE = kwargs['CELL_SIZE']
    TRAIN_SIZE = kwargs['TRAIN_SIZE']

    if kwargs['cell_init'] == 'none':
        cell_node_features = torch.ones(CELL_SIZE).long()
    elif kwargs['cell_init'] == 'pca':
        embedder_mod1 = TruncatedSVD(n_components=100)
        X_train_np = embedder_mod1.fit_transform(input_train_mod1.toarray())
        X_test_np = embedder_mod1.transform(input_test_mod1.toarray())
        cell_node_features = torch.cat([torch.from_numpy(X_train_np), torch.from_numpy(X_test_np)], 0).float()
    if (not kwargs['no_pathway']) and (kwargs['subtask'].find('rna') != -1):
        enhance_graph = construct_pathway_graph(train_mod1, **kwargs)
    else:
        enhance_graph = None

    if kwargs['inductive'] != 'trans':
        u = torch.from_numpy(
            np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        e = torch.from_numpy(input_train_mod1.tocsr().data).float()
        g = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, **kwargs)

        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(np.concatenate(
            [input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        gtest = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, test=True, **kwargs)
        return g, gtest

    else:
        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(np.concatenate(
            [input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        g = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, **kwargs)

        return g


def basic_feature_propagation(dataset, layers, transformed=True, device='cuda'):
    if transformed:
        g1 = construct_basic_feature_graph(dataset.preprocessed_features['mod1_train'],
                                           dataset.preprocessed_features['mod1_test'], device=device)
        g2 = construct_basic_feature_graph(dataset.preprocessed_features['mod2_train'],
                                           dataset.preprocessed_features['mod2_test'], device=device)
    else:
        g1 = construct_basic_feature_graph(dataset.sparse_features()[0], dataset.sparse_features()[2], device=device)
        g2 = construct_basic_feature_graph(dataset.sparse_features()[1], dataset.sparse_features()[3], device=device)

    hcell_mod1 = basic_feature_graph_propagation(g1, layers, device=device)
    hcell_mod2 = basic_feature_graph_propagation(g2, layers, device=device)

    return hcell_mod1, hcell_mod2


def basic_feature_graph_propagation(g, layers=3, alpha=0.5, beta=0.5, cell_init=None, feature_init='id', device='cuda',
                                    verbose=True):

    assert layers > 2, 'Less than two feature graph propagation layers is equivalent to original features.'

    gconv = dglnn.HeteroGraphConv(
        {
            'cell2feature': dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
            'feature2cell': dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
        }, aggregate='sum')

    if feature_init is None:
        feature_X = torch.zeros((g.nodes('feature').shape[0], g.srcdata[cell_init]['cell'].shape[1])).to(device)
    elif feature_init == 'id':
        feature_X = F.one_hot(g.srcdata['id']['feature']).float().to(device)

    if cell_init is None:
        cell_X = torch.zeros(g.nodes('cell').shape[0], feature_X.shape[1]).to(device)
    else:
        cell_X = g.srcdata[cell_init]['cell']

    h = {'feature': feature_X, 'cell': cell_X}
    hcell = []
    for i in range(layers):
        h1 = gconv(
            g, h, mod_kwargs={
                'cell2feature': {
                    'edge_weight': g.edges['cell2feature'].data['weight']
                },
                'feature2cell': {
                    'edge_weight': g.edges['feature2cell'].data['weight']
                }
            })
        if verbose: print(i, 'cell', h['cell'].abs().mean(), h1['cell'].abs().mean())
        if verbose: print(i, 'feature', h['feature'].abs().mean(), h1['feature'].abs().mean())

        h1['feature'] = (h1['feature'] -
                         h1['feature'].mean()) / (h1['feature'].std() if h1['feature'].mean() != 0 else 1)
        h1['cell'] = (h1['cell'] - h1['cell'].mean()) / (h1['cell'].std() if h1['cell'].mean() != 0 else 1)

        h = {
            'feature': h['feature'] * alpha + h1['feature'] * (1 - alpha),
            'cell': h['cell'] * beta + h1['cell'] * (1 - beta)
        }

        h['feature'] = (h['feature'] - h['feature'].mean()) / h['feature'].std()
        h['cell'] = (h['cell'] - h['cell'].mean()) / h['cell'].std()

        hcell.append(h['cell'])

    if verbose: print(hcell[-1].abs().mean())

    return hcell[1:]


##### scGNN create adjacency, likely much overlap with above functions, nested function defs to avoid possible namespace conflicts


def scGNNgenerateAdj(featureMatrix, graphType='KNNgraph', para=None, parallelLimit=0, adjTag=True):
    """Generating edgeList."""

    def calculateKNNgraphDistanceMatrixPairwise(featureMatrix, para):
        r"""KNNgraphPairwise:  measuareName:k Pairwise:5 Minkowski-Pairwise:5:1."""
        measureName = ''
        k = 5
        if para != None:
            parawords = para.split(':')
            measureName = parawords[0]

        distMat = None
        if measureName == 'Pairwise':
            distMat = distance_matrix(featureMatrix, featureMatrix)
            k = int(parawords[1])
        elif measureName == 'Minkowski-Pairwise':
            p = int(parawords[2])
            distMat = minkowski_distance(featureMatrix, featureMatrix, p=p)
            k = int(parawords[1])
        else:
            print('meausreName in KNNgraph does not recongnized')
        edgeList = []

        for i in np.arange(distMat.shape[0]):
            res = distMat[:, i].argsort()[:k]
            for j in np.arange(k):
                edgeList.append((i, res[j]))

        return edgeList

    def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
        r"""
        KNNgraph:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        distanceType incude:
        Distance functions between two numeric vectors u and v. Computing distances over a large collection of vectors is inefficient for these functions. Use pdist for this purpose.
        braycurtis(u, v[, w])	Compute the Bray-Curtis distance between two 1-D arrays.
        canberra(u, v[, w])	Compute the Canberra distance between two 1-D arrays.
        chebyshev(u, v[, w])	Compute the Chebyshev distance.
        cityblock(u, v[, w])	Compute the City Block (Manhattan) distance.
        correlation(u, v[, w, centered])	Compute the correlation distance between two 1-D arrays.
        cosine(u, v[, w])	Compute the Cosine distance between 1-D arrays.
        euclidean(u, v[, w])	Computes the Euclidean distance between two 1-D arrays.
        jensenshannon(p, q[, base])	Compute the Jensen-Shannon distance (metric) between two 1-D probability arrays.
        mahalanobis(u, v, VI)	Compute the Mahalanobis distance between two 1-D arrays.
        minkowski(u, v[, p, w])	Compute the Minkowski distance between two 1-D arrays.
        seuclidean(u, v, V)	Return the standardized Euclidean distance between two 1-D arrays.
        sqeuclidean(u, v[, w])	Compute the squared Euclidean distance between two 1-D arrays.
        wminkowski(u, v, p, w)	Compute the weighted Minkowski distance between two 1-D arrays.
        Distance functions between two boolean vectors (representing sets) u and v. As in the case of numerical vectors, pdist is more efficient for computing the distances between all pairs.
        dice(u, v[, w])	Compute the Dice dissimilarity between two boolean 1-D arrays.
        hamming(u, v[, w])	Compute the Hamming distance between two 1-D arrays.
        jaccard(u, v[, w])	Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays.
        kulsinski(u, v[, w])	Compute the Kulsinski dissimilarity between two boolean 1-D arrays.
        rogerstanimoto(u, v[, w])	Compute the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.
        russellrao(u, v[, w])	Compute the Russell-Rao dissimilarity between two boolean 1-D arrays.
        sokalmichener(u, v[, w])	Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.
        sokalsneath(u, v[, w])	Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.
        yule(u, v[, w])	Compute the Yule dissimilarity between two boolean 1-D arrays.
        hamming also operates over discrete numerical vectors.
        """

        distMat = distance.cdist(featureMatrix, featureMatrix, distanceType)

        edgeList = []

        for i in np.arange(distMat.shape[0]):
            res = distMat[:, i].argsort()[:k]
            for j in np.arange(k):
                edgeList.append((i, res[j]))

        return edgeList

    edgeList = None
    adj = None

    def calculateThresholdgraphDistanceMatrix(featureMatrix, distanceType='euclidean', threshold=0.5):
        r"""Thresholdgraph: Graph with certain threshold."""

        distMat = distance.cdist(featureMatrix, featureMatrix, distanceType)

        edgeList = []

        for i in np.arange(distMat.shape[0]):
            indexArray = np.where(distMat[i, :] > threshold)
            for j in indexArray[0]:
                edgeList.append((i, j))

        return edgeList

    def calculateKNNThresholdgraphDistanceMatrix(featureMatrix, distanceType='cosine', k=10, threshold=0.5):
        r"""Thresholdgraph: KNN Graph with certain threshold."""

        distMat = distance.cdist(featureMatrix, featureMatrix, distanceType)

        edgeList = []

        for i in np.arange(distMat.shape[0]):
            res = distMat[:, i].argsort()[:k]
            for j in np.arange(k - 1):
                if (distMat[i, res[j]] > threshold):
                    edgeList.append((i, res[j]))
            # edgeList.append((i,res[k-1]))

        return edgeList

    def calculateKNNgraphDistanceMatrixML(featureMatrix, distanceType='euclidean', k=10, param=None):
        r"""Thresholdgraph: KNN Graph with Machine Learning based methods
        IsolationForest https://scikit-learn.org/stable/modules/generated/sklearn.ensemb
        le.IsolationForest.html#sklearn.ensemble.IsolationForest."""

        distMat = distance.cdist(featureMatrix, featureMatrix, distanceType)
        edgeList = []

        # parallel: n_jobs=-1 for using all processors
        clf = IsolationForest(behaviour='new', contamination='auto', n_jobs=-1)

        for i in np.arange(distMat.shape[0]):
            res = distMat[i, :].argsort()[:k + 1]
            preds = clf.fit_predict(featureMatrix[res, :])
            for j in np.arange(1, k + 1):
                # weight = 1.0
                if preds[j] == -1:
                    weight = 0.0
                else:
                    weight = 1.0
                # preds[j]==-1 means outliner, 1 is what we want
                edgeList.append((i, res[j], weight))

        return edgeList

    def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10, param=None):
        r"""Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread
        version."""

        edgeList = []
        # Version 1: cost memory, precalculate all dist

        ## distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        ## parallel
        # distMat = pairwise_distances(featureMatrix,featureMatrix, distanceType, n_jobs=-1)

        # for i in np.arange(distMat.shape[0]):
        #     res = distMat[:,i].argsort()[:k+1]
        #     tmpdist = distMat[res[1:k+1],i]
        #     mean = np.mean(tmpdist)
        #     std = np.std(tmpdist)
        #     for j in np.arange(1,k+1):
        #         if (distMat[i,res[j]]<=mean+std) and (distMat[i,res[j]]>=mean-std):
        #             weight = 1.0
        #         else:
        #             weight = 0.0
        #         edgeList.append((i,res[j],weight))

        ## Version 2: for each of the cell, calculate dist, save memory
        p_time = time.time()
        for i in np.arange(featureMatrix.shape[0]):
            if i % 10000 == 0:
                print('Start pruning ' + str(i) + 'th cell, cost ' + str(time.time() - p_time) + 's')
            tmp = featureMatrix[i, :].reshape(1, -1)
            distMat = distance.cdist(tmp, featureMatrix, distanceType)
            res = distMat.argsort()[:k + 1]
            tmpdist = distMat[0, res[0][1:k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            for j in np.arange(1, k + 1):
                # should check, only exclude large outliners
                # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
                if distMat[0, res[0][j]] <= boundary:
                    weight = 1.0
                else:
                    weight = 0.0
                edgeList.append((i, res[0][j], weight))
        return edgeList

    def calculateKNNgraphDistanceMatrixStats(featureMatrix, distanceType='euclidean', k=10, param=None,
                                             parallelLimit=0):
        r"""Thresholdgraph: KNN Graph with stats one-std based methods using parallel
        cores."""
        edgeList = []
        # Get number of availble cores
        USE_CORES = 0
        NUM_CORES = multiprocessing.cpu_count()
        # if no limit, use all cores
        if parallelLimit == 0:
            USE_CORES = NUM_CORES
        # if limit < cores, use limit number
        elif parallelLimit < NUM_CORES:
            USE_CORES = parallelLimit
        # if limit is not valid, use all cores
        else:
            USE_CORES = NUM_CORES
        print('Start Pruning using ' + str(USE_CORES) + ' of ' + str(NUM_CORES) + ' available cores')

        t = time.time()
        # Use number of cpus for top-K finding
        with Pool(USE_CORES) as p:
            # edgeListT = p.map(vecfindK, range(featureMatrix.shape[0]))
            edgeListT = FindKParallel(featureMatrix, distanceType, k).work()

        t1 = time.time()
        print('Pruning succeed in ' + str(t1 - t) + ' seconds')
        flatten = lambda l: [item for sublist in l for item in sublist]
        t2 = time.time()
        edgeList = flatten(edgeListT)
        print('Prune out ready in ' + str(t2 - t1) + ' seconds')

        return edgeList

    class FindKParallel():
        """A class to find K parallel."""

        def __init__(self, featureMatrix, distanceType, k):
            self.featureMatrix = featureMatrix
            self.distanceType = distanceType
            self.k = k

        def vecfindK(self, i):
            """Find topK in paral."""
            edgeList_t = []
            # print('*'+str(i))
            tmp = self.featureMatrix[i, :].reshape(1, -1)
            distMat = distance.cdist(tmp, self.featureMatrix, self.distanceType)
            # print('#'+str(distMat))
            res = distMat.argsort()[:self.k + 1]
            # print('!'+str(res))
            tmpdist = distMat[0, res[0][1:self.k + 1]]
            # print('@'+str(tmpdist))
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            # print('&'+str(boundary))
            for j in np.arange(1, self.k + 1):
                # check, only exclude large outliners
                # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
                if distMat[0, res[0][j]] <= boundary:
                    weight = kernelDistance(distMat[0, res[0][j]])
                    edgeList_t.append((i, res[0][j], weight))
            # print('%'+str(len(edgeList_t)))
            return edgeList_t

        def work(self):
            return Pool().map(self.vecfindK, range(self.featureMatrix.shape[0]))

    def edgeList2edgeDict(edgeList, nodesize):
        graphdict = {}
        tdict = {}

        for edge in edgeList:
            end1 = edge[0]
            end2 = edge[1]
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist

        # check and get full matrix
        for i in range(nodesize):
            if i not in tdict:
                graphdict[i] = []

        return graphdict

    if graphType == 'KNNgraphPairwise':
        edgeList = calculateKNNgraphDistanceMatrixPairwise(featureMatrix, para)
    elif graphType == 'KNNgraph':
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrix(featureMatrix, distanceType=distanceType, k=k)
    elif graphType == 'Thresholdgraph':
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            threshold = float(parawords[1])
        edgeList = calculateThresholdgraphDistanceMatrix(featureMatrix, distanceType=distanceType, threshold=threshold)
    elif graphType == 'KNNgraphThreshold':
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
            threshold = float(parawords[2])
        edgeList = calculateKNNThresholdgraphDistanceMatrix(featureMatrix, distanceType=distanceType, k=k,
                                                            threshold=threshold)
    elif graphType == 'KNNgraphML':
        # with weights!
        # https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
        # https://scikit-learn.org/stable/modules/outlier_detection.html
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrixML(featureMatrix, distanceType=distanceType, k=k)
    elif graphType == 'KNNgraphStats':
        # with weights!
        # with stats, one std is contained
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrixStats(featureMatrix, distanceType=distanceType, k=k,
                                                        parallelLimit=parallelLimit)
    elif graphType == 'KNNgraphStatsSingleThread':
        # with weights!
        # with stats, one std is contained, but only use single thread
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType=distanceType, k=k)
    else:
        print('Should give graphtype')

    if adjTag:
        graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

    return adj, edgeList
