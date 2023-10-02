import dgl
import numpy as np
import torch

from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import WeightedFeaturePCA
from dance.typing import LogLevel, Optional


class CellFeatureGraph(BaseTransform):

    def __init__(self, cell_feature_channel: str, gene_feature_channel: Optional[str] = None, *,
                 mod: Optional[str] = None, normalize_edges: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.cell_feature_channel = cell_feature_channel
        self.gene_feature_channel = gene_feature_channel or cell_feature_channel
        self.mod = mod
        self.normalize_edges = normalize_edges

    def _alternative_construction(self, data):
        # TODO: Try this alternative construction
        x_sparse = data.get_feature(return_type="sparse", channel=self.cell_feature_channel, mod=self.mod)
        g = dgl.bipartite_from_scipy(x_sparse, utype="cell", etype="expression", vtype="feature", eweight_name="weight")
        g = dgl.ToSimple()(g)
        g = dgl.AddSelfLoop(edge_feat_names="weight")(g)
        g = dgl.AddReverse(copy_edata=True)(g)
        g.ndata["weight"] = dgl.nn.EdgeWeightNorm(norm="both")(g, g.ndata["weight"])
        data.data.uns[self.out] = g
        return data

    def __call__(self, data):
        feat = data.get_feature(return_type="default", mod=self.mod)
        num_cells, num_feats = feat.shape

        row, col = np.nonzero(feat)
        edata = np.array(feat[row, col]).ravel()[:, None]
        self.logger.info(f"Number of nonzero entries: {edata.size:,}")
        self.logger.info(f"Nonzero rate = {edata.size / num_cells / num_feats:.1%}")

        row = row + num_feats  # offset by feature nodes
        col, row = np.hstack((col, row)), np.hstack((row, col))  # convert to undirected
        edata = np.vstack((edata, edata))

        # Convert to tensors
        col = torch.LongTensor(col)
        row = torch.LongTensor(row)
        edata = torch.FloatTensor(edata)

        # Initialize cell-gene graph
        g = dgl.graph((row, col))
        g.edata["weight"] = edata
        # FIX: change to feat_id
        g.ndata["cell_id"] = torch.concat((torch.arange(num_feats, dtype=torch.int32),
                                           -torch.ones(num_cells, dtype=torch.int32)))  # yapf: disable
        g.ndata["feat_id"] = torch.concat((-torch.ones(num_feats, dtype=torch.int32),
                                           torch.arange(num_cells, dtype=torch.int32)))  # yapf: disable

        # Normalize edges and add self-loop
        if self.normalize_edges:
            in_deg = g.in_degrees()
            for i in range(g.number_of_nodes()):
                src, dst, eidx = g.in_edges(i, form="all")
                if src.shape[0] > 0:
                    edge_w = g.edata["weight"][eidx]
                    g.edata["weight"][eidx] = in_deg[i] * edge_w / edge_w.sum()
        g.add_edges(g.nodes(), g.nodes(), {"weight": torch.ones(g.number_of_nodes())[:, None]})

        gene_feature = data.get_feature(return_type="torch", channel=self.gene_feature_channel, mod=self.mod,
                                        channel_type="varm")
        cell_feature = data.get_feature(return_type="torch", channel=self.cell_feature_channel, mod=self.mod,
                                        channel_type="obsm")
        g.ndata["features"] = torch.vstack((gene_feature, cell_feature))

        data.data.uns[self.out] = g

        return data


class PCACellFeatureGraph(BaseTransform):

    _DISPLAY_ATTRS = ("n_components", "split_name")

    def __init__(
        self,
        n_components: int = 400,
        split_name: Optional[str] = None,
        *,
        normalize_edges: bool = True,
        feat_norm_mode: Optional[str] = None,
        feat_norm_axis: int = 0,
        mod: Optional[str] = None,
        log_level: LogLevel = "WARNING",
    ):
        super().__init__(log_level=log_level)

        self.n_components = n_components
        self.split_name = split_name
        self.normalize_edges = normalize_edges
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis
        self.mod = mod

    def __call__(self, data):
        WeightedFeaturePCA(self.n_components, self.split_name, feat_norm_mode=self.feat_norm_mode,
                           feat_norm_axis=self.feat_norm_axis, log_level=self.log_level)(data)
        CellFeatureGraph(cell_feature_channel="WeightedFeaturePCA", mod=self.mod, normalize_edges=self.normalize_edges,
                         log_level=self.log_level)(data)
        return data


class CellFeatureBipartiteGraph(BaseTransform):

    def __init__(self, cell_feature_channel: str, *, mod: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.cell_feature_channel = cell_feature_channel
        self.mod = mod

    def __call__(self, data):
        feat = data.get_feature(channel=self.cell_feature_channel, return_type="sparse", mod=self.mod)
        g = dgl.bipartite_from_scipy(feat, utype='cell', etype='cell2feature', vtype='feature', eweight_name='weight')
        g.nodes['cell'].data['id'] = torch.arange(feat.shape[0]).long()
        g.nodes['feature'].data['id'] = torch.arange(feat.shape[1]).long()
        g = dgl.AddReverse(copy_edata=True, sym_new_etype=True)(g)
        if self.mod is None:
            data.data.uns['g'] = g
        else:
            data.data[self.mod].uns['g'] = g
        return data
