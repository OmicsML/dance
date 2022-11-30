import dgl
import numpy as np
import torch

from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import WeightedGenePCA
from dance.typing import LogLevel, Optional


class CellGeneGraph(BaseTransform):

    def __init__(self, cell_feature_channel: str, gene_feature_channel: Optional[str] = None, *,
                 layer: Optional[str] = None, mod: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.cell_feature_channel = cell_feature_channel
        self.gene_feature_channel = gene_feature_channel or cell_feature_channel
        self.layer = layer
        self.mod = mod

    def __call__(self, data):
        feat = data.get_feature(return_type="default", layer=self.layer, mod=self.mod)
        num_cells, num_feats = feat.shape

        row, col = np.nonzero(feat)
        edata = torch.from_numpy(np.array(feat[col, row]).ravel()[:, None])
        size = edata.size()[0]
        self.logger.info(f"Number of nonzero entries: {size:,}")
        self.logger.info(f"Nonzero rate = {size / num_cells / num_feats:.1%}")

        col = col + num_feats  # offset by feature nodes
        col, row = np.hstack((col, row)), np.hstack((row, col))  # convert to undirected
        edata = np.hstack((edata, edata))

        g = dgl.graph((row, col))
        g.edata["weight"] = edata
        g.ndata["id"] = torch.hstack((torch.arange(num_feats,
                                                   dtype=torch.int32), -torch.ones(num_cells, dtype=torch.int32)))

        # Normalize edges and add self-loop
        in_deg = g.in_degrees()
        for i in range(g.number_of_nodes()):
            src, dst, eidx = g.in_edges(i, form="all")
            if src.shape[0] > 0:
                edge_w = g.edata["weight"][eidx]
                g.edata["weight"][eidx] = in_deg[i] * edge_w / edge_w.sum()
        g.add_edges(g.nodes(), g.nodes(), {"weight": torch.ones(g.number_of_nodes())[:, None]})

        gene_feature = data.get_feature(return_type="tensor", channel=self.gene_feature_channel, mod=self.mod)
        cell_feature = data.get_feature(return_type="tensor", channel=self.cell_feature_channel, mod=self.mod)
        g.ndata["features"] = torch.vstack((gene_feature, cell_feature))

        data.data.uns[self.out] = g

        return data


class PCACellGeneGraph(BaseTransform):

    def __init__(self, n_components: int = 400, split_name: Optional[str] = None, *, layer: Optional[str] = None,
                 mod: Optional[str] = None, log_level: LogLevel = "WARNING"):
        super().__init__(log_level=log_level)

        self.n_components = n_components
        self.split_name = split_name

    def __call__(self, data):
        WeightedGenePCA(self.n_components, self.split_name, log_level=self.log_level)(data)
        CellGeneGraph(cell_feature_channel="WeightedGenePCA", layer=self.layer, mod=self.mod,
                      log_level=self.log_level)(data)
        return data
