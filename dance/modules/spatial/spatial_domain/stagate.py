"""Reimplementation of STAGATE.

Extended from https://github.com/QIFEIDKN/STAGATE

Reference
----------
Dong, Kangning, and Shihua Zhang. "Deciphering spatial domains from spatially resolved transcriptomics with an adaptive
graph attention auto-encoder." Nature communications 13.1 (2022): 1-12.

"""
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import mixture
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
from tqdm import tqdm

from dance import logger
from dance.modules.base import BaseClusteringMethod
from dance.transforms import AnnDataTransform, Compose, SetConfig
from dance.transforms.graph import StagateGraph
from dance.typing import Any, LogLevel, Optional
from dance.utils import get_device


class GATConv(MessagePassing):
    """Graph attention layer from Graph Attention Network."""
    _alpha = None

    def __init__(self, in_channels, out_channels, heads: int = 1, concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops=True, bias=True, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None
        self.attentions = None

    def forward(self, x, edge_index, size=None, return_attention_weights=None, attention=True, tied_attention=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in GATConv"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in GATConv"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)

        if tied_attention is None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


class Stagate(torch.nn.Module, BaseClusteringMethod):
    """Stagate class.

    Parameters
    ----------
    hidden_dims
        Hidden dimensions.
    device
        Computation device.

    """

    def __init__(self, hidden_dims, device: str = "auto"):
        super().__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)

        self.device = get_device(device)
        self.to(self.device)

    @staticmethod
    def preprocessing_pipeline(hvg_flavor: str = "seurat_v3", n_top_hvgs: int = 3000, model_name: str = "radius",
                               radius: float = 150, n_neighbors: int = 5, log_level: LogLevel = "INFO"):
        return Compose(
            AnnDataTransform(sc.pp.highly_variable_genes, flavor=hvg_flavor, n_top_genes=n_top_hvgs, subset=True),
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            StagateGraph(model_name, radius=radius, n_neighbors=n_neighbors),
            SetConfig({
                "feature_channel": "StagateGraph",
                "feature_channel_type": "obsp",
                "label_channel": "label",
                "label_channel_type": "obs"
            }),
            log_level=log_level,
        )

    def forward(self, features, edge_index):
        """Forward function for training.

        Parameters
        ----------
        features
            Node features.
        edge_index
            Adjacent matrix.

        Returns
        -------
        Tuple[Tensor, Tensor]
            The second and the forth hidden layerx.

        """
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True, tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4

    def fit(
        self,
        x: np.ndarray,
        edge_index_array: np.ndarray,
        n_epochs: int = 1,
        lr: float = 0.001,
        gradient_clipping: float = 5,
        weight_decay: float = 1e-4,
        num_cluster: int = 7,
    ):
        """Fit function for training.

        Parameters
        ----------
        x
            Input feature.
        edge_index_array
            Edge index (coo representation) as (2 x num_edges) numpy array.
        n_epochs
            Number of epochs.
        lr
            Learning rate.
        gradient_clipping
            Gradient clipping.
        weight_decay
            Weight decay.
        num_cluster
            Number of cluster.

        """
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
        edge_index_tensor = torch.from_numpy(edge_index_array.astype(int)).to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()
        for epoch in tqdm(range(1, n_epochs + 1)):
            optimizer.zero_grad()
            z, out = self(x_tensor, edge_index_tensor)
            loss = F.mse_loss(x_tensor, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clipping)
            optimizer.step()

        self.eval()
        z, _ = self(x_tensor, edge_index_tensor)
        self.rep = z.detach().clone().cpu().numpy()

        logger.info("Start post-processing")
        gm = mixture.GaussianMixture(n_components=num_cluster, covariance_type="tied", warm_start=True, n_init=100,
                                     max_iter=300, reg_covar=1.4663143602030552e-04, random_state=36282,
                                     tol=0.00022187708009762592)
        self.clust_res = gm.fit_predict(x)

    def predict(self, x: Optional[Any] = None):
        """Prediction function.

        Parameters
        ----------
        x
            Not used, for compatibility with :class:`BaseClusteringMethod`.

        """
        return self.clust_res
