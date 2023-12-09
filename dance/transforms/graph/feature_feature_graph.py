import dgl
import dgl.nn as dglnn
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.stats import spearmanr

from dance.transforms.base import BaseTransform
from dance.typing import Any, Dict, Optional
from dance.utils.matrix import dist_to_rbf


class FeatureFeatureGraph(BaseTransform):
    """Feature-feature similarity graph.

    Parameters
    ----------
    threshold
        Edge similarity score threshold.
    positive_only
        Only use positive similarity score if set to ``True``.
    normalize_edges
        Normalize edge weights following GCN if set to ``True``.
    score_func
        Distance function to use, supported options are ``"pearson"``, ``"spearman"``, and ``"rbf"``
    score_func_kwargs
        Optional kwargs passed to the score function, e.g. see :meth:`dance.utils.matrix.dist_to_rbf`.

    """

    _DISPLAY_ATTRS = ("threshold", "positive_only", "normalize_edges", "score_func", "score_func_kwargs")

    def __init__(self, threshold: float = 0.3, *, positive_only: bool = False, normalize_edges: bool = True,
                 score_func="pearson", score_func_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)

        self.threshold = threshold
        self.positive_only = positive_only
        self.normalize_edges = normalize_edges
        self.score_func = score_func
        self.score_func_kwargs = score_func_kwargs or {}

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy")

        # Calculate correlation between features
        if self.score_func == "pearson":
            adj = np.corrcoef(feat.T)
        elif self.score_func == "spearman":
            adj = spearmanr(feat, axis=0)[0]
        elif self.score_func == "rbf":
            norm_vec = np.power(feat, 2).sum(0, keepdims=True)
            # NOTE: need to clip zero due to numerical errors caused by single
            # precision computationa leading to negative values.
            dist_mat = np.sqrt((norm_vec + norm_vec.T - 2 * feat.T @ feat).clip(0))
            adj = dist_to_rbf(dist_mat, **self.score_func_kwargs)
        else:
            raise ValueError(f"Unknown similarity score function {self.score_func!r}, "
                             "supported options are: 'pearson', 'spearman', 'rbf'")

        # Cast type
        feat = feat.astype(np.float32)
        adj = adj.astype(np.float32)

        # Apply threshold
        adj[np.logical_and(adj > -self.threshold, adj < self.threshold)] = 0
        if self.positive_only:
            adj[adj < 0] = 0

        # Initialize graph
        adj_coo = coo_matrix(adj)
        graph_data = (torch.from_numpy(adj_coo.row).int(), torch.from_numpy(adj_coo.col).int())
        g = dgl.graph(graph_data, num_nodes=adj.shape[0])
        g.ndata["feat"] = torch.from_numpy(feat.T)
        g.edata["weight"] = torch.ones(g.num_edges(), dtype=torch.float32)

        # Normalize edges
        if self.normalize_edges:
            norm = dglnn.EdgeWeightNorm()
            norm_edge_weight = norm(g, g.edata["weight"])
            g.edata["weight"] = norm_edge_weight.float()

        data.data.uns[self.out] = g

        return data
