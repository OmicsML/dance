import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import KDTree

import dance.transforms.preprocess
from dance.transforms.base import BaseTransform


class DSTGraph(BaseTransform):
    """DSTG link graph construction.

    The link graph consists of pseudo-spot nodes and real-spot nodes, where the psudo-spots are generated from
    reference data with known cell-type portions. The real-spot nodes are from the data. The linkage, i.e., edges
    are derived based on mutual nearest neighbor in the cononical correlation analysis embedding space.

    Parameters
    ----------
    k_filter : int
        Number of k-nearest neighbors to keep in the final graph.
    num_cc : int
        Number of dimensions to use in the concanical correlation analysis.
    ref_split : str
        Name of the reference data split, i.e., the pseudo-spot data.
    inf_split : str
        Name of the inference data split, i.e., the real-spot data.

    """

    _DISPLAY_ATTRS = ("k_filter", "num_cc", "ref_split", "inf_split")

    def __init__(self, k_filter=200, num_cc=30, *, ref_split: str = "train", inf_split: str = "test", **kwargs):
        super().__init__(**kwargs)

        self.k_filter = k_filter
        self.num_cc = num_cc
        self.ref_split = ref_split
        self.inf_split = inf_split

    def __call__(self, data):
        x_ref = data.get_feature(return_type="numpy", split_name=self.ref_split)
        x_inf = data.get_feature(return_type="numpy", split_name=self.inf_split)

        adj = compute_dstg_adj(x_ref, x_inf, k_filter=self.k_filter, num_cc=self.num_cc)
        data.data.obsp[self.out] = adj

        return data


def compute_dstg_adj(pseudo_st_scale, real_st_scale, k_filter=300, num_cc=30):
    num_ref = len(pseudo_st_scale)
    num_inf = len(real_st_scale)
    num_tot = num_ref + num_inf

    pseudo_st_df = pd.DataFrame(pseudo_st_scale.T, columns=range(num_ref))
    real_st_df = pd.DataFrame(real_st_scale.T, columns=range(num_ref, num_tot))
    graph = construct_link_graph(pseudo_st_df, real_st_df, k_filter, num_cc)

    # Combine the spot index for pseudo (find) and real (index) into the graph node index
    index = np.concatenate((np.arange(num_ref), -np.ones(num_inf)))
    find = np.concatenate((-np.ones(num_ref), np.arange(num_inf)))
    index_map = {j: i for i, j in enumerate(index) if j >= 0}
    find_map = {j: i for i, j in enumerate(find) if j >= 0}

    edge_sets_dict = {i: {i} for i in range(num_tot)}  # initialize graph with self-loops
    for _, (i, j) in graph.iterrows():
        col, row = index_map[i], find_map[j]
        edge_sets_dict[row].add(col)
        edge_sets_dict[col].add(row)

    adj = nx.to_scipy_sparse_array(nx.from_dict_of_lists(edge_sets_dict))
    adj_normalized = preprocess_adj(adj)

    return adj_normalized


def construct_link_graph(pseudo_st_df, real_st_df, k_filter=200, num_cc=30):
    cell_embedding, loading = dance.transforms.preprocess.ccaEmbed(pseudo_st_df, real_st_df, num_cc=num_cc)
    norm_embedding = dance.transforms.preprocess.l2norm(mat=cell_embedding[0])
    spots1 = pseudo_st_df.columns
    spots2 = real_st_df.columns
    neighbor = knn(cell_embedding=norm_embedding, spots1=spots1, spots2=spots2, k=30)
    mnn_edges = mnn(neighbors=neighbor, colnames=cell_embedding[0].index, num=5)
    select_genes = dance.transforms.preprocess.selectTopGenes(Loadings=loading, dims=range(num_cc), DimGenes=100,
                                                              maxGenes=200)
    Mat = pd.concat((pseudo_st_df, real_st_df), axis=1)
    graph = filter_edge(edges=mnn_edges, neighbors=neighbor, mats=Mat, features=select_genes, k_filter=k_filter)

    return graph


def filter_edge(edges, neighbors, mats, features, k_filter):
    nn_spots1, nn_spots2 = neighbors[4:6]
    mat1 = mats.loc[features, nn_spots1].T
    mat2 = mats.loc[features, nn_spots2].T
    cn_data1 = dance.transforms.preprocess.l2norm(mat1)
    cn_data2 = dance.transforms.preprocess.l2norm(mat2)
    nn = query_knn(data=cn_data2.loc[nn_spots2], query=cn_data1.loc[nn_spots1], k=k_filter)
    ind = [j in nn[1][i] for _, (i, j) in edges.iterrows()]
    filtered_edges = edges[ind].copy().reset_index(drop=True)
    return filtered_edges


def preprocess_adj(adj):
    """Symmetrically normalize the adjacency matrix with an addition of identity."""
    # Question: isn't this addition of self-loop redundant with the initialization?
    adj = sp.csr_matrix(adj + sp.eye(adj.shape[0]))
    d_inv_sqrt = sp.diags(1 / np.sqrt(adj.sum(1).A.ravel()))
    adj_normalized = d_inv_sqrt.dot(adj).dot(d_inv_sqrt).tocoo()
    return adj_normalized


def query_knn(data, k, query=None):
    tree = KDTree(data)
    dist, ind = tree.query(data if query is None else query, k)
    return dist, ind


def knn(cell_embedding, spots1, spots2, k):
    embedding_spots1 = cell_embedding.loc[
        spots1,
    ]
    embedding_spots2 = cell_embedding.loc[
        spots2,
    ]
    nnaa = query_knn(embedding_spots1, k=k + 1)
    nnbb = query_knn(embedding_spots2, k=k + 1)
    nnab = query_knn(data=embedding_spots2, k=k, query=embedding_spots1)
    nnba = query_knn(data=embedding_spots1, k=k, query=embedding_spots2)
    return nnaa, nnab, nnba, nnbb, spots1, spots2


def mnn(neighbors, colnames, num):
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
