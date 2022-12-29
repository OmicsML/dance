import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import KDTree

import dance.transforms.preprocess
from dance.transforms.base import BaseTransform


class DSTGraph(BaseTransform):
    pass


def stAdjConstruct(st_scale, st_label, adj_data, k_filter=1):
    st_scale_dfs = [adata.to_df().T for adata in st_scale]
    graph = stLinkGraphConstruct(st_scale_dfs, st_label, k_filter)

    data_train1, data_val1, data_test1, labels, lab_data2 = adj_data

    fake1 = np.array([-1] * len(lab_data2.index))
    index1 = np.concatenate((data_train1.index, fake1, data_val1.index, data_test1.index)).flatten()
    fake2 = np.array([-1] * len(data_train1))
    fake3 = np.array([-1] * (len(data_val1) + len(data_test1)))
    find1 = np.concatenate((fake2, np.array(lab_data2.index), fake3)).flatten()

    row1 = [np.where(find1 == graph.iloc[i, 1])[0][0] for i in range(len(graph))]
    col1 = [np.where(index1 == graph.iloc[i, 0])[0][0] for i in range(len(graph))]
    adj = defaultdict(list)  # default value of int is 0
    for i in range(len(labels)):
        adj[i].append(i)
    for i in range(len(row1)):
        adj[row1[i]].append(col1[i])
        adj[col1[i]].append(row1[i])

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))
    adj_normalized = preprocess_adj(adj)

    return adj_normalized


def stLinkGraphConstruct(st_scale, st_label, k_filter):
    if (n := len(st_scale)) == 1:
        combine = pd.Series([(0, 0)])
    else:
        combine = pd.Series(itertools.combinations(range(n), 2))

    link = stLinkGraph(scale_list=st_scale, combine=combine, k_filter=k_filter)
    assert len(link) == 1, "Default DSTG graph construct only uses the firs two expression matrices."
    graph = link[0].iloc[:, :2].reset_index(drop=True)

    return graph


def stLinkGraph(scale_list, combine, k_filter=200, num_cc=30):
    all_edges = []
    for i, j in combine:
        scale_data1 = scale_list[i]
        scale_data2 = scale_list[j]

        cell_embedding, loading = dance.transforms.preprocess.ccaEmbed(scale_data1, scale_data2, num_cc=num_cc)
        norm_embedding = dance.transforms.preprocess.l2norm(mat=cell_embedding[0])
        spots1 = scale_data1.columns
        spots2 = scale_data2.columns
        neighbor = KNN(cell_embedding=norm_embedding, spots1=spots1, spots2=spots2, k=30)
        mnn_edges = MNN(neighbors=neighbor, colnames=cell_embedding[0].index, num=5)
        select_genes = dance.transforms.preprocess.selectTopGenes(Loadings=loading, dims=range(num_cc), DimGenes=100,
                                                                  maxGenes=200)
        Mat = pd.concat([scale_data1, scale_data2], axis=1)
        edges = filterEdge(edges=mnn_edges, neighbors=neighbor, mats=Mat, features=select_genes, k_filter=k_filter)
        edges[["Dataset1", "Dataset2"]] = i, j
        all_edges.append(edges)
    return all_edges


def filterEdge(edges, neighbors, mats, features, k_filter):
    nn_spots1, nn_spots2 = neighbors[4:6]
    mat1 = mats.loc[features, nn_spots1].T
    mat2 = mats.loc[features, nn_spots2].T
    cn_data1 = dance.transforms.preprocess.l2norm(mat1)
    cn_data2 = dance.transforms.preprocess.l2norm(mat2)
    nn = kNN(data=cn_data2.loc[nn_spots2, ], query=cn_data1.loc[nn_spots1, ], k=k_filter)
    position = [
        np.where(edges.loc[:, "spot2"][x] == nn[1][edges.loc[:, "spot1"][x], ])[0] for x in range(edges.shape[0])
    ]
    nps = np.concatenate(position, axis=0)
    fedge = edges.iloc[nps, ]
    return (fedge)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for scGCN model and conversion to tuple
    representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized  # sparse_to_tuple(adj_normalized)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def kNN(data, k, query=None):
    tree = KDTree(data)
    dist, ind = tree.query(data if query is None else query, k)
    return dist, ind


def KNN(cell_embedding, spots1, spots2, k):
    embedding_spots1 = cell_embedding.loc[spots1, ]
    embedding_spots2 = cell_embedding.loc[spots2, ]
    nnaa = kNN(embedding_spots1, k=k + 1)
    nnbb = kNN(embedding_spots2, k=k + 1)
    nnab = kNN(data=embedding_spots2, k=k, query=embedding_spots1)
    nnba = kNN(data=embedding_spots1, k=k, query=embedding_spots2)
    return nnaa, nnab, nnba, nnbb, spots1, spots2


def MNN(neighbors, colnames, num):
    max_nn = np.array([neighbors[1][1].shape[1], neighbors[2][1].shape[1]])
    if ((num > max_nn).any()):
        num = np.min(max_nn)
        # convert cell name to neighbor index
    spots1 = colnames
    # spots2 = colnames
    nn_spots1 = neighbors[4]
    # nn_spots2 = neighbors[5]
    cell1_index = [list(nn_spots1).index(i) for i in spots1 if (nn_spots1 == i).any()]
    # cell2_index = [list(nn_spots2).index(i) for i in spots2 if (nn_spots2 == i).any()]
    ncell = range(neighbors[1][1].shape[0])
    ncell = np.array(ncell)[np.in1d(ncell, cell1_index)]
    # initialize a list
    mnn_cell1 = [None] * (len(ncell) * 5)
    mnn_cell2 = [None] * (len(ncell) * 5)
    idx = -1
    for cell in ncell:
        neighbors_ab = neighbors[1][1][cell, 0:5]
        mutual_neighbors = np.where(neighbors[2][1][neighbors_ab, 0:5] == cell)[0]
        for i in neighbors_ab[mutual_neighbors]:
            idx = idx + 1
            mnn_cell1[idx] = cell
            mnn_cell2[idx] = i
    mnn_cell1 = mnn_cell1[0:(idx + 1)]
    mnn_cell2 = mnn_cell2[0:(idx + 1)]
    import pandas as pd
    mnns = pd.DataFrame(np.column_stack((mnn_cell1, mnn_cell2)))
    mnns.columns = ["spot1", "spot2"]
    return mnns
