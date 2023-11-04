import dgl
import dgl.nn as dglnn
import torch
from scipy.sparse import coo_matrix
from torch.nn import functional as F
from dance.utils.matrix import pairwise_distance

from dance.transforms.base import BaseTransform


class FeatureFeatureGraph(BaseTransform):

    def __init__(self, threshold: float = 0.3, *, normalize_edges: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.threshold = threshold
        self.normalize_edges = normalize_edges

    def __call__(self, data,dist_func_id=0):
        feat = data.get_feature(return_type="torch")

        # Calculate correlation between features
        #TODO get more distance options
        corr = torch.from_numpy(pairwise_distance(feat.T.detach().numpy(),dist_func_id=dist_func_id))
        corr_sub = F.threshold(corr, self.threshold, 0) - F.threshold(-corr, self.threshold, 0)
        corr_coo = coo_matrix(corr_sub)

        # Initialize graph
        graph_data = (torch.from_numpy(corr_coo.row).int(), torch.from_numpy(corr_coo.col).int())
        g = dgl.graph(graph_data, num_nodes=corr.shape[0])
        g.ndata["feat"] = feat.T
        g.edata["weight"] = torch.ones(g.num_edges()).float()

        # Normalize edges
        if self.normalize_edges:
            norm = dglnn.EdgeWeightNorm()
            norm_edge_weight = norm(g, g.edata["weight"])
            g.edata["weight"] = norm_edge_weight.float()

        data.data.uns[self.out] = g

        return data
