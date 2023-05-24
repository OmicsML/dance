from time import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from igraph import Graph
from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.preprocessing import minmax_scale
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

from dance import logger
from dance.typing import Any, Optional
from dance.utils import get_device


class ScGNN2:

    # FIX: unwrap args
    def __init__(self, args, device: str = "auto"):
        self.args = args
        self.device = get_device(device)

    def fit(self, x: np.ndarray):
        args = self.args
        epochs = self.args.total_epoch

        # Set up the program
        param = dict()
        param["device"] = self.device
        param["dataloader_kwargs"] = {} if self.device == "cpu" else {"num_workers": 1, "pin_memory": True}
        param["tik"] = time()
        logger.info(f"Using device: {param['device']}")

        trs_mat = np.zeros_like(x)
        ccc_graph = None

        # Main program starts here
        logger.info("Pre EM runs")
        param["epoch_num"] = 0
        param["total_epoch"] = epochs
        param["n_feature_orig"] = x.shape[1]
        param["x_dropout"] = x
        x_process = x.copy()

        x_embed, x_feature_recon, model_state = feature_AE_handler(x_process, trs_mat, args, param)
        graph_embed, CCC_graph_hat, edgeList, adj = graph_AE_handler(x_embed, ccc_graph, args, param)

        logger.info("Entering main loop")
        for i in range(epochs):
            logger.info(f"\n==========> scGNN Epoch {i+1}/{epochs} <==========")
            param["epoch_num"] = i + 1
            param["feature_embed"] = x_embed
            param["graph_embed"] = graph_embed

            cluster_labels, cluster_lists_of_idx = clustering_handler(edgeList, args, param)
            param["impute_regu"] = graph_celltype_regu_handler(adj, cluster_labels)
            x_imputed = cluster_AE_handler(x_feature_recon, trs_mat, cluster_lists_of_idx, args, param, model_state)

            x_embed, x_feature_recon, model_state = feature_AE_handler(x_imputed, trs_mat, args, param, model_state)
            graph_embed, CCC_graph_hat, edgeList, adj = graph_AE_handler(x_embed, ccc_graph, args, param)

        self.x_imputed = x_imputed

    def predict(self, x: Optional[Any] = None) -> np.ndarray:
        return self.x_imputed


# ---------------------------
# args.seed
# args.clustering_embed
# args.clustering_louvain_only
# args.clustering_method
# args.clustering_use_flexible_k


# param["clustering_embed"]
# param["feature_embed"]
# param["graph_embed"]
# param["k_float"]
# ---------------------------
# def clustering_handler(edgeList, args, param, metrics):
def clustering_handler(edgeList, args, param):
    logger.info("Start Clustering")

    louvain_only = args.clustering_louvain_only
    # use_flexible_k = args.clustering_use_flexible_k
    # all_ct_count = metrics.metrics["cluster_count"]
    clustering_embed = args.clustering_embed
    clustering_method = args.clustering_method
    # avg_factor = 0.95

    if clustering_embed == "graph":
        embed = param["graph_embed"]
    elif clustering_embed == "feature":
        embed = param["feature_embed"]
    elif clustering_embed == "both":
        feature_embed_norm = normalizer(param["feature_embed"], base=param["graph_embed"], axis=0)
        embed = np.concatenate((param["graph_embed"], feature_embed_norm), axis=1)
    else:
        logger.error("clustering_embed argument not recognized, using graph embed")
        embed = param["graph_embed"]

    param["clustering_embed"] = embed

    # edgeList = (cell_i, cell_a), (cell_i, cell_b), ...
    listResult, size = generateLouvainCluster(edgeList)
    k_Louvain = len(np.unique(listResult))
    logger.info(f" Louvain clusters count: {k_Louvain}")

    resolution = 0.8 if embed.shape[0] < 2000 else 0.5  # based on num of cells
    k_resolution = max(k_Louvain * resolution, 2)

    # Seems like len(all_ct_count) will alawys be 1, since 'cluster_count' is never updated
    # if use_flexible_k or len(all_ct_count) == 1:
    #     param["k_float"] = k_resolution
    # else:
    #     param["k_float"] = avg_factor * param["k_float"] + (1-avg_factor) * k_resolution
    param["k_float"] = k_resolution

    k = round(param["k_float"])
    logger.info(f" Adjusted clusters count: {k}")

    if not louvain_only:
        if clustering_method == "KMeans":
            # (n_samples,) Index of the cluster each sample belongs to
            listResult = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(embed)
        elif clustering_method == "AffinityPropagation":
            listResult = AffinityPropagation(random_state=args.seed).fit_predict(embed)

    if len(set(listResult)) > 30 or len(set(listResult)) <= 1:
        logger.info(f" Stopping: Number of clusters is {len(set(listResult))}")
        listResult = trimClustering(listResult, minMemberinCluster=5, maxClusterNumber=30)

    logger.info(f"Total Cluster Number: {len(set(listResult))}")
    return cluster_output_handler(listResult)  # tuple{"ct_list", "lists_of_idx"}


def generateLouvainCluster(edgeList):
    """Louvain Clustering using igraph."""
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(W.tolist(), mode="undirected", attr="weight", loops=False)

    louvain_partition = graph.community_multilevel(weights=graph.es["weight"], return_levels=False)
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


def cluster_output_handler(listResult):
    clusterIndexList = []
    for i in range(len(set(listResult))):
        clusterIndexList.append([])
    for i in range(len(listResult)):
        clusterIndexList[listResult[i]].append(i)

    return listResult, clusterIndexList


def trimClustering(listResult, minMemberinCluster=5, maxClusterNumber=30):
    """If the clustering numbers larger than certain number, use this function to trim.

    May have better solution

    """
    numDict = {}
    for item in listResult:
        if item not in numDict:
            numDict[item] = 0
        else:
            numDict[item] = numDict[item] + 1

    size = len(set(listResult))
    changeDict = {}
    for item in range(size):
        if numDict[item] < minMemberinCluster or item >= maxClusterNumber:
            changeDict[item] = ""

    count = 0
    for item in listResult:
        if item in changeDict:
            listResult[count] = maxClusterNumber
        count += 1

    return listResult


# ------------------------------
# args.feature_AE_batch_size
# args.feature_AE_epoch
# args.feature_AE_learning_rate
# args.feature_AE_regu_strength
# args.feature_AE_dropout_prob
# args.feature_AE_concat_prev_embed


# param["dataloader_kwargs"]
# param["device"]
# param["epoch_num"]
# param["feature_embed"]
# param["graph_embed"]
# param["impute_regu"]
# param["n_feature_orig"]
# param["x_dropout"]
# ------------------------------
def feature_AE_handler(X, TRS, args, param, model_state=None):
    logger.info("Starting Feature AE")

    batch_size = args.feature_AE_batch_size
    total_epoch = args.feature_AE_epoch[param["epoch_num"] > 0]
    learning_rate = args.feature_AE_learning_rate
    regu_strength = args.feature_AE_regu_strength
    masked_prob = args.feature_AE_dropout_prob
    concat_prev_embed = args.feature_AE_concat_prev_embed

    if concat_prev_embed and param["epoch_num"] > 0:
        if concat_prev_embed == "graph":
            prev_embed = normalizer(param["graph_embed"], base=X, axis=0)
        elif concat_prev_embed == "feature":
            prev_embed = param["feature_embed"]
        else:
            logger.info(" feature_AE_concat_prev_embed argument not recognized, not using any previous embed")
            prev_embed = None
        X = np.concatenate((X, prev_embed), axis=1)

    X_dataset = ExpressionDataset(X)
    X_loader = DataLoader(X_dataset, batch_size=batch_size, **param["dataloader_kwargs"])
    TRS = torch.from_numpy(TRS).type(torch.FloatTensor)

    feature_AE = Feature_AE(dim=X.shape[1]).to(param["device"])
    optimizer = optim.Adam(feature_AE.parameters(), lr=learning_rate)

    impute_regu = None
    if param["epoch_num"] > 0:
        # Load Graph and Celltype Regu
        adjdense, celltypesample = param["impute_regu"]
        adjsample = torch.from_numpy(adjdense).type(torch.FloatTensor).to(param["device"])
        celltypesample = torch.from_numpy(celltypesample).type(torch.FloatTensor).to(param["device"])
        # Load x_dropout as regu
        x_dropout = torch.from_numpy(param["x_dropout"]).type(torch.FloatTensor).to(param["device"])
        impute_regu = {"graph_regu": adjsample, "celltype_regu": celltypesample, "x_dropout": x_dropout}

        if concat_prev_embed and param["epoch_num"] > 1:
            feature_AE.load_state_dict(model_state["model_concat"])
        elif not concat_prev_embed:
            feature_AE.load_state_dict(model_state["model"])

    _, X_embed, X_recon = train_handler(model=feature_AE, train_loader=X_loader, optimizer=optimizer, TRS=TRS,
                                        total_epoch=total_epoch, impute_regu=impute_regu, regu_type=["LTMG", "noregu"],
                                        regu_strength=regu_strength, masked_prob=masked_prob, param=param)

    if concat_prev_embed and param["epoch_num"] > 0:
        checkpoint = {
            "model_concat": feature_AE.state_dict(),
            "optimizer_concat": optimizer.state_dict(),
            "model": model_state["model"],
            "optimizer": model_state["optimizer"]
        }
    else:
        checkpoint = {"model": feature_AE.state_dict(), "optimizer": optimizer.state_dict()}

    X_embed_out = X_embed.detach().cpu().numpy()
    X_recon_out = X_recon.detach().cpu().numpy()
    X_recon_out = X_recon_out[:, :param["n_feature_orig"]]

    return X_embed_out, X_recon_out, checkpoint  # cell * {gene, embedding}


class Feature_AE(nn.Module):
    '''
    Autoencoder for dimensional reduction
    Args:
        x: Tensor, mini-batch
        dim: int, feature dimension
    Return:
        self.decode(z): reconstructed input
        z: feature encoding
    '''

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
        return F.relu(self.fc4(h3))

        # h3 = torch.sigmoid(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return z, self.decode(z)


class Graph_AE(nn.Module):

    def __init__(self, dim, embedding_size, gat_dropout=0, multi_heads=2, gat_hid_embed=64):
        super().__init__()
        self.gat = GAT(num_of_layers=2, num_heads_per_layer=[multi_heads, multi_heads],
                       num_features_per_layer=[dim, gat_hid_embed, embedding_size], dropout=gat_dropout)

        self.gc1 = GraphConvolution(dim, 32, 0, act=F.relu)
        self.gc2 = GraphConvolution(32, embedding_size, 0, act=lambda x: x)
        self.gc3 = GraphConvolution(32, embedding_size, 0, act=lambda x: x)

        self.decode = InnerProductDecoder(0, act=lambda x: x)

    def encode_gat(self, in_nodes_features, edge_index):
        return self.gat((in_nodes_features, edge_index))

    def encode_gae(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, in_nodes_features, edge_index, encode=False, use_GAT=True):
        # [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        gae_info = None

        if use_GAT:
            out_nodes_features = self.encode_gat(in_nodes_features, edge_index)[0]
        else:
            gae_info = self.encode_gae(in_nodes_features, edge_index)
            out_nodes_features = self.reparameterize(*gae_info)

        recon_graph = self.decode(out_nodes_features)
        return out_nodes_features, gae_info, recon_graph


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


class Cluster_AE(Feature_AE):

    def __init__(self, dim):
        super().__init__(dim)


class GCNModelVAE(nn.Module):

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


def edgeList2edgeIndex(edgeList):
    result = [[i[0], i[1]] for i in edgeList]
    return result


# --------------------------
# args.graph_AE_use_GAT
# args.graph_AE_learning_rate
# args.graph_AE_epoch
# args.graph_AE_embedding_size
# args.graph_AE_concat_prev_embed
# args.graph_AE_normalize_embed
# args.graph_AE_GAT_dropout
# args.graph_AE_neighborhood_factor
# args.graph_AE_retain_weights


# param["epoch_num"]
# param["graph_embed"]
# --------------------------
def graph_AE_handler(X_embed, CCC_graph, args, param):
    logger.info("Starting Graph AE")

    use_GAT = args.graph_AE_use_GAT
    learning_rate = args.graph_AE_learning_rate
    total_epoch = args.graph_AE_epoch
    embedding_size = args.graph_AE_embedding_size
    concat_prev_embed = args.graph_AE_concat_prev_embed
    normalize_embed = args.graph_AE_normalize_embed
    gat_dropout = args.graph_AE_GAT_dropout
    neighborhood_factor = args.graph_AE_neighborhood_factor
    retain_weights = args.graph_AE_retain_weights

    if concat_prev_embed and param["epoch_num"] > 0:
        graph_embed = param["graph_embed"]
        graph_embed_norm = normalizer(graph_embed, base=X_embed, axis=0)
        X_embed = np.concatenate((X_embed, graph_embed_norm), axis=1)

    if normalize_embed == "sum1":
        zDiscret = normalize_features_dense(X_embed)
    elif normalize_embed == "binary":
        zDiscret = 1.0 * (X_embed > np.mean(X_embed, axis=0))
    else:
        zDiscret = X_embed

    adj, adj_train, edgeList = feature2adj(X_embed, neighborhood_factor, retain_weights)
    adj_norm = preprocess_graph(adj_train)
    adj_label = (adj_train + sp.eye(adj_train.shape[0])).toarray()

    # Prepare matrices
    if use_GAT:
        edgeIndex = edgeList2edgeIndex(edgeList)
        edgeIndex = np.array(edgeIndex).T
        CCC_graph_edge_index = torch.from_numpy(edgeIndex).type(torch.LongTensor).to(param["device"])
    else:
        CCC_graph_edge_index = adj_norm.to(param["device"])

        pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
        norm = adj_train.shape[0] * adj_train.shape[0] / \
            float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)

    X_embed_normalized = torch.from_numpy(zDiscret).type(torch.FloatTensor).to(param["device"])
    CCC_graph = torch.from_numpy(adj_label).type(torch.FloatTensor).to(param["device"])

    graph_AE = Graph_AE(X_embed.shape[1], embedding_size, gat_dropout, args.gat_multi_heads,
                        args.gat_hid_embed).to(param["device"])
    optimizer = optim.Adam(graph_AE.parameters(), lr=learning_rate)

    for epoch in range(total_epoch):
        graph_AE.train()
        optimizer.zero_grad()

        embed, gae_info, recon_graph = graph_AE(X_embed_normalized, CCC_graph_edge_index, use_GAT=use_GAT)

        if use_GAT:
            loss = loss_function(preds=recon_graph, labels=CCC_graph)
        else:
            loss = gae_loss_function(preds=recon_graph, labels=CCC_graph, mu=gae_info[0], logvar=gae_info[1],
                                     n_nodes=X_embed.shape[0], norm=norm, pos_weight=pos_weight)

        # Backprop and Update
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        logger.info(f"Epoch: {epoch+1}/{total_epoch}, Current loss: {cur_loss:.4f}")

    embed_out = embed.detach().cpu().numpy()
    recon_graph_out = recon_graph.detach().cpu().numpy()

    return embed_out, recon_graph_out, edgeList, adj


def gae_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def loss_function(preds, labels):
    return F.binary_cross_entropy_with_logits(preds, labels)


def normalize_features_dense(node_features_dense):
    assert isinstance(node_features_dense, np.ndarray), f"Expected np matrix got {type(node_features_dense)}."

    # The goal is to make feature vectors normalized (sum equals 1), but since some feature vectors are all 0s
    # in those cases we"d have division by 0 so I set the min value (via np.clip) to 1.
    # Note: 1 is a neutral element for division i.e. it won"t modify the feature vector
    return node_features_dense / np.clip(node_features_dense.sum(1, keepdims=True), a_min=1, a_max=None)


def convert_adj_to_edge_index(adjacency_matrix):
    """"""
    assert isinstance(adjacency_matrix, np.ndarray), f"Expected NumPy array got {type(adjacency_matrix)}."
    height, width = adjacency_matrix.shape
    assert height == width, f"Expected square shape got = {adjacency_matrix.shape}."

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    # active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] > 0:
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(edge_index).transpose()  # change shape from (N,2) -> (2,N)


def feature2adj(X_embed, neighborhood_factor, retain_weights):
    neighborhood_size_temp = neighborhood_factor if neighborhood_factor > 1 else round(X_embed.shape[0] *
                                                                                       neighborhood_factor)
    neighborhood_size = neighborhood_size_temp - \
        1 if neighborhood_size_temp == X_embed.shape[0] else neighborhood_size_temp

    edgeList = calculateKNNgraphDistanceMatrixStatsSingleThread(X_embed, k=neighborhood_size)

    if retain_weights:
        G = nx.DiGraph()
        G.add_weighted_edges_from(edgeList)
        adj_return = nx.adjacency_matrix(G)
    else:
        graphdict = edgeList2edgeDict(edgeList, X_embed.shape[0])
        adj_return = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

    adj = adj_return.copy()

    # Clear diagonal elements (no self loop)
    adj_train = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_train.eliminate_zeros()

    return adj_return, adj_train, edgeList


def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType="euclidean", k=10):
    r"""Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread
    version."""

    edgeList = []

    for i in np.arange(featureMatrix.shape[0]):
        tmp = featureMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, featureMatrix, distanceType)
        res = distMat.argsort()[:k + 1]
        for j in np.arange(1, k + 1):
            weight = 1 / (distMat[0, res[0][j]] + 1e-16)
            edgeList.append((i, res[0][j], weight))

    return edgeList


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


def graph_celltype_regu_handler(adj, cluster_labels):
    adjdense = sp.csr_matrix.todense(adj)
    adjdense = normalize_cell_cell_matrix(adjdense)

    celltypesample = generateCelltypeRegu(cluster_labels)
    celltypesample = normalize_cell_cell_matrix(celltypesample)

    return adjdense, celltypesample


def normalize_cell_cell_matrix(x):
    avg_factor = 1 / np.ma.sum(x, axis=1).reshape((x.shape[0], -1))
    avg_factor = np.ma.filled(avg_factor, fill_value=0)
    avg_mtx = np.tile(avg_factor, [1, len(x)])
    return avg_mtx * x


def generateCelltypeRegu(listResult):
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


class ExpressionDataset(Dataset):

    def __init__(self, X=None, transform=None):
        """
        Args:
            X : ndarray (dense) or list of lists (sparse) [cell * gene]
            transform (callable, optional): apply transform function if not none
        """
        self.X = X  # [cell * gene]
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]  # of cell

    def __getitem__(self, idx):

        # Get sample (one cell)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        # Convert to Tensor
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class ClusterDataset(ExpressionDataset):

    def __init__(self, X=None, transform=None):
        super().__init__(X, transform)


def normalizer(X, base, axis=0):
    upper = np.quantile(base, q=0.9)
    lower = np.quantile(base, q=0.1)
    if upper != lower:
        normalized = minmax_scale(X, feature_range=(lower, upper), axis=axis)
    else:
        max = np.quantile(base, q=1)
        min = np.quantile(base, q=0)
        normalized = minmax_scale(X, feature_range=(min, max), axis=axis)

    return normalized


# ---------------------------
# args.cluster_AE_batch_size
# args.cluster_AE_epoch
# args.cluster_AE_learning_rate
# args.cluster_AE_regu_strength
# args.cluster_AE_dropout_prob


# param["dataloader_kwargs"]
# param["device"]
# param["impute_regu"]
# param["x_dropout"]
# ---------------------------
def cluster_AE_handler(X_recon, TRS, clusterIndexList, args, param, model_state):
    logger.info("Starting Cluster AE")

    batch_size = args.cluster_AE_batch_size
    total_epoch = args.cluster_AE_epoch
    learning_rate = args.cluster_AE_learning_rate
    regu_strength = args.cluster_AE_regu_strength
    masked_prob = args.cluster_AE_dropout_prob

    # Initialize an empty matrix for storing the results
    reconNew = np.zeros_like(X_recon)
    reconNew = torch.from_numpy(reconNew).type(torch.FloatTensor).to(param["device"])

    # Load Graph and Celltype Regu
    adjdense, celltypesample = param["impute_regu"]
    adjsample = torch.from_numpy(adjdense).type(torch.FloatTensor)
    celltypesample = torch.from_numpy(celltypesample).type(torch.FloatTensor)

    x_dropout = torch.from_numpy(param["x_dropout"]).type(torch.FloatTensor)
    TRS = torch.from_numpy(TRS).type(torch.FloatTensor)

    for i, clusterIndex in enumerate(clusterIndexList):
        logger.info(f"Training cluster {i+1}/{len(clusterIndexList)} -> size = {len(clusterIndex)}")

        # Define separate models for each cell type, they should not share weights
        cluster_AE = Cluster_AE(dim=X_recon.shape[1]).to(param["device"])
        optimizer = optim.Adam(cluster_AE.parameters(), lr=learning_rate)

        # Load weights from Feature AE to save training time
        cluster_AE.load_state_dict(model_state["model"])

        adjsample_ct = adjsample[clusterIndex][:, clusterIndex].to(param["device"])
        celltypesample_ct = celltypesample[clusterIndex][:, clusterIndex].to(
            param["device"])  # this is just an all 1"s square matrix
        x_dropout_ct = x_dropout[clusterIndex].to(param["device"])

        impute_regu = {"graph_regu": adjsample_ct, "celltype_regu": celltypesample_ct, "x_dropout": x_dropout_ct}

        reconUsage = X_recon[clusterIndex]
        scDataInter = ClusterDataset(reconUsage)
        X_loader = DataLoader(scDataInter, batch_size=batch_size, **param["dataloader_kwargs"])

        Cluster_orig, Cluster_embed, Cluster_recon = train_handler(model=cluster_AE, train_loader=X_loader,
                                                                   optimizer=optimizer, TRS=TRS,
                                                                   total_epoch=total_epoch, impute_regu=impute_regu,
                                                                   regu_type=[None,
                                                                              "Celltype"], regu_strength=regu_strength,
                                                                   masked_prob=masked_prob, param=param)

        for i, row in enumerate(clusterIndex):
            reconNew[row] = Cluster_recon[i, :]

        # empty cuda cache
        del Cluster_orig
        del Cluster_embed
        torch.cuda.empty_cache()

    recon_out = reconNew.detach().cpu().numpy()

    return recon_out  # cell * gene


class GAT(torch.nn.Module):
    """The most interesting and hardest implementation is implementation #3. Imp1 and
    imp2 differ in subtle details but are basically the same thing.

    So I'll focus on imp #3 in this notebook.

    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, 'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights)
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(*gat_layers, )

    # data is just a (in_nodes_features, edge_index) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    """Implementation #3 was inspired by PyTorch Geometric:
    https://github.com/rusty1s/pytorch_geometric.

    But, it's hopefully much more readable! (and of similar performance)

    """

    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0  # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1  # attention head dim

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
            scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """As the fn name suggest it does softmax over the neighborhoods. Example: say
        we have 5 nodes in a graph. Two of them 1, 2 are connected to node 3. If we want
        to calculate the representation for node 3 we should take into account feature
        vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3 in
        scores_per_edge variable, this function will calculate attention scores like
        this: 1-3/(1-3+2-3+3-3) (where 1-3 is overloaded notation it represents the edge
        1-3 and its (exp) score) and similarly for 2-3 and 3-3 i.e. for this
        neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """Lifts i.e. duplicates certain vectors depending on the edge index.

        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """The reason we're using Glorot (aka Xavier uniform) initialization is because
        it's a default TF initialization:
        https://stackoverflow.com/questions/37350131/what-is-the-default-variable-
        initializer-in-tensorflow.

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sparse_mx.tocoo().astype(np.float64)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.DoubleTensor(indices, values, shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ---------------------------
# param["epoch_num"]
# param["total_epoch"]
# param["device"]
# ---------------------------
def train_handler(
    model,
    train_loader,
    optimizer,
    TRS,
    total_epoch,
    regu_strength,
    masked_prob,
    param,
    regu_type,
    impute_regu,
):
    """EMFlag indicates whether in EM processes.

    If in EM, use regulized-type parsed from program entrance,
    Otherwise, noregu
    taskType: celltype or imputation

    """

    if len(regu_type) == 2:
        current_regu_type = regu_type[param["epoch_num"] > 0]
    else:  # len(regu_type) == 3:
        if param["epoch_num"] == 0:  # Pre EM
            current_regu_type = regu_type[0]
        elif param["epoch_num"] == param["total_epoch"]:  # Last epoch
            current_regu_type = regu_type[2]
        else:
            current_regu_type = regu_type[1]  # Non-final EM epochs

    for epoch in range(total_epoch):
        model.train()
        train_loss = 0

        for batch_idx, (data, dataindex) in enumerate(train_loader):  # data is Tensor of shape [batch * gene]

            # Send data and regulation matrix to device
            data = data.type(torch.FloatTensor).to(param["device"])
            data_masked = F.dropout(data, p=masked_prob)
            if impute_regu is not None:
                impute_regu["LTMG_regu"] = TRS[dataindex, :].to(param["device"])
            else:
                impute_regu = {"LTMG_regu": TRS[dataindex, :].to(param["device"])}

            optimizer.zero_grad()

            # Reconstructed batch and encoding layer as outputs
            z, recon_batch = model.forward(data_masked)

            # Calculate loss
            loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), regulationMatrix=impute_regu,
                                       regu_strength=regu_strength, regularizer_type=current_regu_type, param=param)

            if current_regu_type == "Celltype":
                l1 = 0.0
                l2 = 0.0
                for p in model.parameters():
                    l1 = l1 + p.abs().sum()
                    l2 = l2 + p.pow(2).sum()
                loss = loss + 1 * l1 + 0 * l2

            # Backprop and Update
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Grow recon_batch, data, z at each epoch, while printing train loss
            if batch_idx == 0:
                recon_batch_all = recon_batch
                data_all = data
                z_all = z
            else:
                recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
                data_all = torch.cat((data_all, data), 0)
                z_all = torch.cat((z_all, z), 0)

        logger.info(f"Epoch: {epoch+1}/{total_epoch}, Average loss: {train_loss / len(train_loader.dataset):.4f}")

    return data_all, z_all, recon_batch_all


def loss_function_graph(recon_x, x, regulationMatrix=None, regularizer_type="noregu", regu_strength=0.9,
                        reduction="sum", param=None):
    """Regularized by the graph information Reconstruction + KL divergence losses summed
    over all elements and batch."""
    if regularizer_type in ["LTMG", "Celltype"]:
        x.requires_grad = True

    BCE = F.mse_loss(recon_x, x, reduction=reduction)  # [cell batch * gene]

    if regularizer_type == "noregu":
        loss = BCE

    elif regularizer_type == "LTMG":
        loss = ((1 - regu_strength) * BCE + regu_strength *
                (F.mse_loss(recon_x, x, reduction="none") * regulationMatrix["LTMG_regu"]).sum())

    elif regularizer_type == "Celltype":

        regulationMatrix["x_dropout"].requires_grad = True

        nonzero_regu = (regulationMatrix["x_dropout"] -
                        recon_x[:, :param["n_feature_orig"]])[regulationMatrix["x_dropout"].nonzero(as_tuple=True)]
        # [cell*cell] @ [cell*gene] replacing individual cell expressions
        # (i.e. each row) with the sum of expressions of the connected neighbors
        graph_regu = regulationMatrix["graph_regu"] @ F.mse_loss(recon_x, x, reduction="none")
        # [cell*cell] @ [cell*gene] replacing individual cell expressions
        # (i.e. each row) with the sum of expressions within the cell type
        # to which a cell belongs
        celltype_norm = regulationMatrix["celltype_regu"] @ F.mse_loss(recon_x, x, reduction="none")

        loss = 0.3 * BCE + torch.norm(nonzero_regu) + 0.3 * graph_regu.sum() + 0.1 * celltype_norm.sum()

    return loss
