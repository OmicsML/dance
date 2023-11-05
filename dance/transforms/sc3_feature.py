import math
import dgl
import dgl.nn as dglnn
import torch
from scipy.sparse import coo_matrix
from torch.nn import functional as F
from dance.utils.matrix import pairwise_distance
from sklearn.decomposition import PCA,TruncatedSVD
from dance.transforms.base import BaseTransform
import numpy as np
from sklearn.cluster import KMeans

def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1/np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

class SC3Feature(BaseTransform):

    def __init__(self, threshold: float = 0.3, *, normalize_edges: bool = True, n_cluster:int=3,**kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.normalize_edges = normalize_edges
        self.n_cluster=n_cluster
        self.choices=None
    def __call__(self, data,d=None):
        feat = data.get_feature(return_type="numpy")
        num_cells=feat.shape[0]
        if d is None:
            d=math.ceil(num_cells*0.07)-math.floor(num_cells*0.04)
        if d>15:
            self.choices=sorted(np.random.choice(range(d),15,replace=False))
        else:
            self.choices=list(range(d))
        y_len=feat.shape[0]
        sc3_mats=[]
        for i in range(3):
            corr = torch.from_numpy(pairwise_distance(np.array(feat).astype(np.float32),dist_func_id=i))
            sc3_mat=corr.numpy()
            mat_pca = PCA(n_components=y_len)
            sc3_mats.append(mat_pca.fit_transform(sc3_mat)[:,self.choices])
            sc3_mats.append(normalized_laplacian(sc3_mat)[:,self.choices])
        sim_matrix_all=[]
        for sc3_mat in sc3_mats:
            for i in range(len(self.choices)):
                sim_matrix=np.identity(y_len)
                y_pred = KMeans(n_clusters=self.n_cluster, random_state=9).fit_predict(sc3_mat[:,0:i+1])
                for i in range(y_len):
                    for j in range(i+1,y_len):
                        y1=y_pred[i]
                        y2=y_pred[j]
                        if (y1==y2):
                            sim_matrix[i][j]=1
                            sim_matrix[j][i]=1
                sim_matrix_all.append(sim_matrix)
        sim_matrix_all=np.array(sim_matrix_all)
        sim_matrix_mean=np.mean(sim_matrix_all,axis=0)
        data.data.uns[self.out]=sim_matrix_mean
        return data

