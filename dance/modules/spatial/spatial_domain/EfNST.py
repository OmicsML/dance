"""Created on Tue Jan 23 18:54:08 2024.

@author: lenovo

"""
import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances


#class Aug:
# def __init__(self, data,spatial_k=50, spatial_type="KDTree"):
#     if spatial_type != "KDTree":
#         raise ValueError("Invalid spatial_type. Supported types are: 'KDTree'.")
#     self.data = data
#     self.spatial_k = spatial_k
#     self.spatial_type = spatial_type
# def cal_spatial_weight(data,spatial_k = 50,spatial_type = "KDTree",):
# 	from sklearn.neighbors import KDTree
# 	if spatial_type == "KDTree":
# 		tree = KDTree(data, leaf_size=2)
# 		_, indices = tree.query(data, k=spatial_k + 1)
# 	indices = indices[:, 1:]
# 	spatial_weight = np.zeros((data.shape[0], data.shape[0]))
# 	for i in range(indices.shape[0]):
# 		ind = indices[i]
# 		for j in ind:
# 			spatial_weight[i][j] = 1
# 	return spatial_weight
def cal_spatial_weight(
    data,
    spatial_k=50,
    spatial_type="KDTree",
):
    from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
    if spatial_type == "NearestNeighbors":
        nbrs = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    elif spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    indices = indices[:, 1:]
    spatial_weight = np.zeros((data.shape[0], data.shape[0]))
    for i in range(indices.shape[0]):
        ind = indices[i]
        for j in ind:
            spatial_weight[i][j] = 1
    return spatial_weight


def cal_gene_weight(data, n_components=50, gene_dist_type="cosine"):

    pca = PCA(n_components=n_components)
    if isinstance(data, np.ndarray):
        data_pca = pca.fit_transform(data)
    elif isinstance(data, csr_matrix):
        data = data.toarray()
        data_pca = pca.fit_transform(data)
    gene_correlation = 1 - pairwise_distances(data_pca, metric=gene_dist_type)
    return gene_correlation


def cal_weight_matrix(adata, platform="Visium", pd_dist_type="euclidean", md_dist_type="cosine",
                      gb_dist_type="correlation", n_components=50, no_morphological=True, spatial_k=30,
                      spatial_type="KDTree", verbose=False):
    if platform == "Visium":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs["array_row"]
        array_col = adata.obs["array_col"]
        rate = 3
        reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
        reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
        physical_distance = pairwise_distances(adata.obs[["imagecol", "imagerow"]], metric=pd_dist_type)
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
        physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
    else:
        physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k=spatial_k, spatial_type=spatial_type)

    gene_counts = adata.X.copy()
    gene_correlation = cal_gene_weight(data=gene_counts, gene_dist_type=gb_dist_type, n_components=n_components)
    del gene_counts
    if verbose:
        adata.obsm["gene_correlation"] = gene_correlation
        adata.obsm["physical_distance"] = physical_distance

    if platform == 'Visium':
        morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
        morphological_similarity[morphological_similarity < 0] = 0
        if verbose:
            adata.obsm["morphological_similarity"] = morphological_similarity
        adata.obsm["weights_matrix_all"] = (physical_distance * gene_correlation * morphological_similarity)
        if no_morphological:
            adata.obsm["weights_matrix_nomd"] = (gene_correlation * physical_distance)
    else:
        adata.obsm["weights_matrix_nomd"] = (gene_correlation * physical_distance)
    return adata


def find_adjacent_spot(adata, use_data="raw", neighbour_k=4, weights='weights_matrix_all', verbose=False):
    if use_data == "raw":
        if isinstance(adata.X, (csr_matrix, np.ndarray)):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            gene_matrix = adata.X.values
        else:
            raise ValueError(f"""{type(adata.X)} is not a valid type.""")
    else:
        gene_matrix = adata.obsm[use_data]
    weights_matrix = adata.obsm[weights]
    weights_list = []
    final_coordinates = []
    for i in range(adata.shape[0]):
        if weights == "physical_distance":
            current_spot = adata.obsm[weights][i].argsort()[-(neighbour_k + 3):][:(neighbour_k + 2)]
        else:
            current_spot = adata.obsm[weights][i].argsort()[-neighbour_k:][:neighbour_k - 1]
        spot_weight = adata.obsm[weights][i][current_spot]
        spot_matrix = gene_matrix[current_spot]
        if spot_weight.sum() > 0:
            spot_weight_scaled = spot_weight / spot_weight.sum()
            weights_list.append(spot_weight_scaled)
            spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
            spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
        else:
            spot_matrix_final = np.zeros(gene_matrix.shape[1])
            weights_list.append(np.zeros(len(current_spot)))
        final_coordinates.append(spot_matrix_final)
    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = np.array(weights_list)
    return adata


def augment_gene_data(adata, Adj_WT=0.2):
    adjacent_gene_matrix = adata.obsm["adjacent_data"].astype(float)
    if isinstance(adata.X, np.ndarray):
        augment_gene_matrix = adata.X + Adj_WT * adjacent_gene_matrix
    elif isinstance(adata.X, csr_matrix):
        augment_gene_matrix = adata.X.toarray() + Adj_WT * adjacent_gene_matrix
    adata.obsm["augment_gene_data"] = augment_gene_matrix
    del adjacent_gene_matrix
    return adata


def augment_adata(adata, platform="Visium", pd_dist_type="euclidean", md_dist_type="cosine", gb_dist_type="correlation",
                  n_components=50, no_morphological=False, use_data="raw", neighbour_k=4, weights="weights_matrix_all",
                  Adj_WT=0.2, spatial_k=30, spatial_type="KDTree"):
    adata = cal_weight_matrix(
        adata,
        platform=platform,
        pd_dist_type=pd_dist_type,
        md_dist_type=md_dist_type,
        gb_dist_type=gb_dist_type,
        n_components=n_components,
        no_morphological=no_morphological,
        spatial_k=spatial_k,
        spatial_type=spatial_type,
    )
    adata = find_adjacent_spot(adata, use_data=use_data, neighbour_k=neighbour_k, weights=weights)
    adata = augment_gene_data(
        adata,
        Adj_WT=Adj_WT,
    )
    return adata


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:28:11 2024

@author: lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):

            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values, )
                return type("Literal_", (Literal, ), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass


_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]


#class VisiumDataProcessor:
def read_Visium(path, genome=None, count_file='filtered_feature_bc_matrix.h5', library_id=None, load_images=True,
                quality='hires', image_path=None):
    adata = sc.read_visium(
        path,
        genome=genome,
        count_file=count_file,
        library_id=library_id,
        load_images=load_images,
    )
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = (adata.obsm["spatial"])
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


class Refiner:

    def __init__(self, shape="hexagon"):
        self.shape = shape
        self.pred_df = None
        self.dis_df = None

    def fit(self, sample_id, pred, dis):
        self.pred_df = pd.DataFrame({"pred": pred}, index=sample_id)
        self.dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)

    def get_neighbors(self, index, num_neighbors):
        distances = self.dis_df.loc[index, :].sort_values()
        return distances.index[1:num_neighbors + 1]

    def majority_vote(self, predictions):
        counts = np.bincount(predictions)
        return np.argmax(counts)

    def refine(sample_id, pred, dis, shape="hexagon"):
        refined_pred = []
        pred = pd.DataFrame({"pred": pred}, index=sample_id)
        dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
        if shape == "hexagon":
            num_nbs = 6
        elif shape == "square":
            num_nbs = 4
        for i in range(len(sample_id)):
            index = sample_id[i]
            dis_tmp = dis_df.loc[index, :].sort_values()
            nbs = dis_tmp[0:num_nbs + 1]
            nbs_pred = pred.loc[nbs.index, "pred"]
            self_pred = pred.loc[index, "pred"]
            v_c = nbs_pred.value_counts()
            if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
                refined_pred.append(v_c.idxmax())
            else:
                refined_pred.append(self_pred)
        return refined_pred


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:36:44 2024

@author: lenovo
"""
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import BatchNorm, Sequential
from torch_sparse import SparseTensor


class graph:

    def __init__(
        self,
        data,
        rad_cutoff,
        k,
        distType='euclidean',
    ):
        super().__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]

    def graph_computing(self):
        dist_list = ["euclidean", "cosine"]
        graphList = []
        if self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(self.data.shape[0])
                         for j in range(indices.shape[1])]
        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList = [(node_idx, indices[j]) for node_idx in range(self.data.shape[0])
                         for j in np.where(A[node_idx] == 1)[0]]
        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(indices.shape[0])
                         for j in range(indices[node_idx].shape[0]) if distances[node_idx][j] > 0]
        return graphList

    def List2Dict(self, graphList):
        graphdict = {}
        tdict = {}
        for end1, end2 in graphList:
            tdict[end1] = ""
            tdict[end2] = ""
            graphdict.setdefault(end1, []).append(end2)
        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []
        return graphdict

    def mx2SparseTensor(self, mx):
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.mx2SparseTensor(adj_normalized)

    def main(self):
        adj_mtx = self.graph_computing()
        graph_dict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
        adj_pre = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
        adj_pre.eliminate_zeros()
        adj_norm = self.preprocess_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)
        graph_dict = {"adj_norm": adj_norm, "adj_label": adj_label, "norm_value": norm}
        return graph_dict

    def combine_graph_dicts(self, dict_1, dict_2):
        tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
        graph_dict = {
            "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
            "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
            "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
        }
        return graph_dict


class EFNST_model(nn.Module):

    def __init__(self, input_dim, Conv_type='ResGatedGraphConv', linear_encoder_hidden=[50, 20],
                 linear_decoder_hidden=[50, 60], conv_hidden=[32, 8], p_drop=0.1, dec_cluster_n=15, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = 0.8
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        current_encoder_dim = self.input_dim
        self.encoder = nn.Sequential()
        for le in range(len(linear_encoder_hidden)):
            self.encoder.add_module(
                f'encoder_L{le}',
                buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate, self.p_drop))
            current_encoder_dim = linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(
                f'decoder_L{ld}',
                buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate, self.p_drop))
            current_decoder_dim = self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}',
                                buildNetwork(self.linear_decoder_hidden[-1], self.input_dim, "sigmoid", p_drop))
        if self.Conv_type == "ResGatedGraphConv":
            from torch_geometric.nn import ResGatedGraphConv
            self.conv = Sequential('x, edge_index', [
                (ResGatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (ResGatedGraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (ResGatedGraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        self.dc = InnerProductDecoder(p_drop)
        self.cluster_layer = Parameter(
            torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1] + self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def target_distribution(self, target):
        weight = (target**2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def EfNST_loss(self, decoded, x, preds, labels, mu, logvar, n_nodes, norm, mask=None, MSE_WT=10, KLD_WT=0.1):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return MSE_WT * mse_loss + KLD_WT * (bce_logits_loss + KLD)

    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat([feat_x, gnn_z], dim=1)
        de_feat = self.decoder(z)
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer)**2, dim=2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return z, mu, logvar, de_feat, q, feat_x, gnn_z


def buildNetwork(in_features, out_features, activate="relu", p_drop=0.0):
    layers = [
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
    ]
    if activate == "relu":
        layers.append(nn.ELU())
    elif activate == "sigmoid":
        layers.append(nn.Sigmoid())
    if p_drop > 0:
        layers.append(nn.Dropout(p_drop))
    return nn.Sequential(*layers)


class InnerProductDecoder(nn.Module):

    def __init__(self, dropout, act=torch.sigmoid):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GradientReverseLayer(torch.autograd.Function):

    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x) * 1.0

    def backward(ctx, grad_output):
        return (grad_output * -1 * ctx.weight), None


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:03:53 2024

@author: lenovo
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn.decomposition import PCA
from torch.autograd import Variable
from tqdm import tqdm


class Image_Feature:

    def __init__(
        self,
        adata,
        pca_components=50,
        cnnType='efficientnet-b0',
        verbose=False,
        seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def efficientNet_model(self):
        efficientnet_versions = {
            'efficientnet-b0': 'efficientnet-b0',
            'efficientnet-b1': 'efficientnet-b1',
            'efficientnet-b2': 'efficientnet-b2',
            'efficientnet-b3': 'efficientnet-b3',
            'efficientnet-b4': 'efficientnet-b4',
            'efficientnet-b5': 'efficientnet-b5',
            'efficientnet-b6': 'efficientnet-b6',
            'efficientnet-b7': 'efficientnet-b7',
        }
        if self.cnnType in efficientnet_versions:
            model_version = efficientnet_versions[self.cnnType]
            cnn_pretrained_model = EfficientNet.from_pretrained(model_version)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(f"{self.cnnType} is not a valid EfficientNet type.")
        return cnn_pretrained_model

    def Extract_Image_Feature(self, ):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAutocontrast(),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
            transforms.RandomInvert(),
            transforms.RandomAdjustSharpness(random.uniform(0, 1)),
            transforms.RandomSolarize(random.uniform(0, 1)),
            transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
            transforms.RandomErasing()
        ]
        img_to_tensor = transforms.Compose(transform_list)
        feat_df = pd.DataFrame()
        model = self.efficientNet_model()
        model.eval()
        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run the function image_crop first")
        for spot, slice_path in self.adata.obs['slices_path'].items():
            spot_slice = Image.open(slice_path)
            spot_slice = spot_slice.resize((224, 224))
            spot_slice = np.asarray(spot_slice, dtype="int32")
            spot_slice = spot_slice.astype(np.float32)
            tensor = img_to_tensor(spot_slice)
            tensor = tensor.resize_(1, 3, 224, 224)
            tensor = tensor.to(self.device)
            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy().ravel()
            feat_df[spot] = result_npy
            feat_df = feat_df.copy()
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata


def image_crop(adata, save_path, library_id=None, crop_size=50, target_size=224, verbose=False, quality='hires'):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
        adata.uns["spatial"][library_id]["use_quality"] = quality
    image = adata.uns["spatial"][library_id]["images"][adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    tile_names = []
    with tqdm(total=len(adata), desc="Tiling image", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.ANTIALIAS)  #####
            tile.resize((target_size, target_size))  ######
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print("generate tile at location ({}, {})".format(str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)
    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata


# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:27:06 2024

@author: lenovo
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import distance
from sklearn.metrics import calinski_harabasz_score


class run():

    def __init__(
        self,
        save_path="./",
        pre_epochs=1000,
        epochs=500,
        pca_n_comps=200,
        linear_encoder_hidden=[32, 20],
        linear_decoder_hidden=[32],
        conv_hidden=[32, 8],
        verbose=True,
        platform='Visium',
        cnnType='efficientnet-b0',
        Conv_type='ResGatedGraphConv',
        p_drop=0.01,
        dec_cluster_n=20,
        n_neighbors=15,
        min_cells=3,
        grad_down=5,
        KL_WT=100,
        MSE_WT=10,
        KLD_WT=0.1,
        Domain_WT=1,
        use_gpu=True,
    ):
        self.save_path = save_path
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.pca_n_comps = pca_n_comps
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.conv_hidden = conv_hidden
        self.verbose = verbose
        self.platform = platform
        self.cnnType = cnnType
        self.Conv_type = Conv_type
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        self.n_neighbors = n_neighbors
        self.min_cells = min_cells
        self.platform = platform
        self.grad_down = grad_down
        self.KL_WT = KL_WT
        self.MSE_WT = MSE_WT
        self.KLD_WT = KLD_WT
        self.Domain_WT = Domain_WT
        self.use_gpu = use_gpu

    def _get_adata(
        self,
        data_path,
        data_name,
        verbose=True,
    ):
        if self.platform == 'Visium':
            adata = read_Visium(os.path.join(data_path, data_name))
        save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', f'{data_name}'))
        save_path_image_crop.mkdir(parents=True, exist_ok=True)
        adata = image_crop(adata, save_path=save_path_image_crop)
        adata = Image_Feature(adata, pca_components=self.pca_n_comps, cnnType=self.cnnType).Extract_Image_Feature()
        if verbose:
            save_data_path = Path(os.path.join(self.save_path, f'{data_name}'))
            save_data_path.mkdir(parents=True, exist_ok=True)
            adata.write(os.path.join(save_data_path, f'{data_name}.h5ad'), compression="gzip")
        return adata

    def _get_graph(
        self,
        data,
        distType="Radius",
        k=12,
        rad_cutoff=150,
    ):
        graph_dict = graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff).main()
        print("Step 2: Graph computing!")
        return graph_dict

    def _get_augment(
        self,
        adata,
        Adj_WT=0.2,
        neighbour_k=4,
        weights="weights_matrix_all",
        spatial_k=30,
    ):
        adata_augment = augment_adata(
            adata,
            Adj_WT=Adj_WT,
            neighbour_k=neighbour_k,
            platform=self.platform,
            weights=weights,
            spatial_k=spatial_k,
        )
        print("Step 1: Augment Gene!")
        return adata_augment

    def _optimize_cluster(
            self,
            adata,
            resolution_range=(0.1, 2.5, 0.01),
    ):
        resolutions = np.arange(*resolution_range)
        scores = [
            calinski_harabasz_score(adata.X,
                                    sc.tl.leiden(adata, resolution=r).obs["leiden"]) for r in resolutions
        ]
        cl_opt_df = pd.DataFrame({"resolution": resolutions, "score": scores})
        best_resolution = cl_opt_df.loc[cl_opt_df["score"].idxmax(), "resolution"]
        return best_resolution

    def _priori_cluster(self, adata, n_domains=7):
        for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == n_domains:
                break
        return res

    def _get__dataset_adata(
        self,
        data_path,
        data_name,
        character="spatial",
        verbose=False,
        Adj_WT=0.2,
        neighbour_k=4,
        weights="weights_matrix_all",
        spatial_k=30,
        distType="Radius",
        k=12,
        rad_cutoff=150,
    ):
        adata = self._get_adata(data_path=data_path, data_name=data_name, verbose=verbose)
        adata = self._get_augment(adata, Adj_WT=Adj_WT, neighbour_k=neighbour_k, weights=weights, spatial_k=spatial_k)
        graph_dict = self._get_graph(adata.obsm[character], distType=distType, k=k, rad_cutoff=rad_cutoff)
        self.data_name = data_name
        if self.verbose:
            print("Step 1: Augment Gene !")
            print("Step 2: Graph computing !")
        return adata, graph_dict

    def _fit(
        self,
        adata,
        graph_dict,
        domains=None,
        dim_reduction=True,
        pretrain=True,
        save_data=False,
    ):
        print("Task sucessful, please wait")
        if self.platform == "Visium":
            adata.X = adata.obsm["augment_gene_data"].astype(float)
            if dim_reduction:
                sc.pp.filter_genes(adata, min_cells=self.min_cells)
                adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
                adata_X = sc.pp.log1p(adata_X)
                adata_X = sc.pp.scale(adata_X)
                concat_X = sc.pp.pca(adata_X, n_comps=self.pca_n_comps)
            else:
                sc.pp.filter_genes(adata, min_cells=self.min_cells)
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
                sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)
                sc.pp.log1p(adata)
                concat_X = adata[:, adata.var['highly_variable']].X
        else:
            concat_X = adata.obsm["augment_gene_data"]
        EfNST_model = EFNST_model(
            input_dim=concat_X.shape[1],
            Conv_type=self.Conv_type,
            linear_encoder_hidden=self.linear_encoder_hidden,
            linear_decoder_hidden=self.linear_decoder_hidden,
            conv_hidden=self.conv_hidden,
            p_drop=self.p_drop,
            dec_cluster_n=self.dec_cluster_n,
        )
        if domains is None:
            EfNST_training = TrainingConfig(
                concat_X,
                graph_dict,
                EfNST_model,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                KL_WT=self.KL_WT,
                MSE_WT=self.MSE_WT,
                KLD_WT=self.KLD_WT,
                Domain_WT=self.Domain_WT,
                use_gpu=self.use_gpu,
            )
        if pretrain:
            EfNST_training.fit()
        else:
            EfNST_training.pretrain(grad_down=self.grad_down)
        EfNST_embedding, _ = EfNST_training.process()
        if self.verbose:
            print("Step 3: Training Done!")
        adata.obsm["EfNST_embedding"] = EfNST_embedding
        return adata

    def _get_cluster_data(
        self,
        adata,
        n_domains,
        priori=True,
    ):
        sc.pp.neighbors(adata, use_rep='EfNST_embedding', n_neighbors=self.n_neighbors)
        if priori:
            res = self._priori_cluster(adata, n_domains=n_domains)
        else:
            res = self._optimize_cluster(adata)
        sc.tl.leiden(adata, key_added="EfNST_domain", resolution=res)
        adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
        refined_pred = Refiner.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["EfNST_domain"].tolist(),
                                      dis=adj_2d, shape="hexagon")
        adata.obs["EfNST"] = refined_pred
        return adata


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:10:01 2024

@author: lenovo
"""

import igraph as ig
import leidenalg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
from sklearn.cluster import KMeans
from torch.autograd import Variable


class TrainingConfig:

    def __init__(self, pro_data, G_dict, model, pre_epochs, epochs, corrupt=0.001, lr=5e-4, weight_decay=1e-4,
                 domains=None, KL_WT=100, MSE_WT=10, KLD_WT=0.1, Domain_WT=1, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.pro_data = pro_data
        self.data = torch.FloatTensor(pro_data.copy()).to(self.device)
        self.adj = G_dict['adj_norm'].to(self.device)
        self.adj_label = G_dict['adj_label'].to(self.device)
        self.norm = G_dict['norm_value']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=weight_decay)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.num_spots = self.data.shape[0]
        self.dec_tol = 0
        self.KL_WT = KL_WT
        self.q_stride = 20
        self.MSE_WT = MSE_WT
        self.KLD_WT = KLD_WT
        self.Domain_WT = Domain_WT
        self.corrupt = corrupt
        self.domains = torch.from_numpy(domains).to(self.device) if domains is not None else domains

    def masking_noise(data, frac):
        data_noise = data.clone()
        rand = torch.rand(data.size())
        data_noise[rand < frac] = 0
        return data_noise

    def pretrain(self, grad_down=5):
        for epoch in range(self.pre_epochs):
            inputs_corr = TrainingConfig.masking_noise(self.data, self.corrupt)
            inputs_coor = inputs_corr.to(self.device)
            self.model.train()
            self.optimizer.zero_grad()
            if self.domains is not None:
                z, mu, logvar, de_feat, _, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor), self.adj)
                preds = self.model.model.dc(z)
            else:
                z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                preds = self.model.dc(z)
            loss = self.model.EfNST_loss(
                decoded=de_feat,
                x=self.data,
                preds=preds,
                labels=self.adj_label,
                mu=mu,
                logvar=logvar,
                n_nodes=self.num_spots,
                norm=self.norm,
                mask=self.adj_label,
                MSE_WT=self.MSE_WT,
                KLD_WT=self.KLD_WT,
            )
            if self.domains is not None:
                loss_function = nn.CrossEntropyLoss()
                Domain_loss = loss_function(domain_pred, self.domains)
                loss += Domain_loss * self.Domain_WT
            else:
                loss = loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_down)
            self.optimizer.step()

    def process(self):
        self.model.eval()
        if self.domains is None:
            z, _, _, _, q, _, _ = self.model(self.data, self.adj)
        else:
            z, _, _, _, q, _, _, _ = self.model(self.data, self.adj)
        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()
        return z, q

    def save_and_load_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])

    def fit(
        self,
        cluster_n=20,
        clusterType='leiden',
        leiden_resolution=1.0,
        pretrain=True,
    ):
        if pretrain:
            self.pretrain()
            pre_z, _ = self.process()
        if clusterType == 'KMeans' and cluster_n is not None:  # 使用K均值算法进行聚类，且聚类数目已知
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
        elif clusterType == 'Leiden':
            if cluster_n is None:
                g = ig.Graph()
                g.add_vertices(pre_z.shape[0])
            for i in range(pre_z.shape[0]):
                for j in range(i + 1, pre_z.shape[0]):
                    g.add_edge(i, j, weight=np.linalg.norm(pre_z[i] - pre_z[j]))
            partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                                 resolution_parameter=leiden_resolution)
            y_pred_last = np.array(partition.membership)
            unique_clusters = np.unique(y_pred_last)
            cluster_centers_ = np.array([pre_z[y_pred_last == cluster].mean(axis=0) for cluster in unique_clusters])
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
        else:
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        if epoch % self.q_stride == 0:
            _, q = self.process()
            q = self.target_distribution(torch.Tensor(q).clone().detach())
            y_pred = q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            self.y_pred_last = np.copy(y_pred)
            if epoch > 0 and delta_label < self.dec_tol:
                return False
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        inputs_coor = self.data.to(self.device)
        if self.domains is None:
            z, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
            preds = self.model.dc(z)
        else:
            z, mu, logvar, de_feat, out_q, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor), self.adj)
            loss_function = nn.CrossEntropyLoss()
            Domain_loss = loss_function(domain_pred, self.domains)
            preds = self.model.model.dc(z)
            loss_EfNST = self.model.EfNST_loss(decoded=de_feat, x=self.data, preds=preds, labels=self.adj_label, mu=mu,
                                               logvar=logvar, n_nodes=self.num_spots, norm=self.norm,
                                               mask=self.adj_label, MSE_WT=self.MSE_WT, KLD_WT=self.KLD_WT)
            loss_KL = F.KL_div(out_q.log(), q.to(self.device))
            if self.domains is None:
                loss = self.KL_WT * loss_KL + loss_EfNST
            else:
                loss = self.KL_WT * loss_KL + loss_EfNST + Domain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
