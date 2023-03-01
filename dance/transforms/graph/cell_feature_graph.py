import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import WeightedFeaturePCA
from dance.typing import LogLevel, Optional
from dgl import AddReverse
import dgl.nn.pytorch as dglnn


#### TODO: Let's try this:
#### g = dgl.bipartite_from_scipy(x_sparse, utype='cell', etype='expression', vtype='feature', eweight_name='weight')
#### g = ToSimple()(g)
#### g = AddSelfLoop(edge_feat_names='weight')(g)
#### g = AddReverse(copy_edata=True)(g)
#### g.ndata['weight'] = EdgeWeightNorm(norm='both')(g, g.ndata['weight'])
class CellFeatureGraph(BaseTransform):

    def __init__(self, cell_feature_channel: str, gene_feature_channel: Optional[str] = None, *,
                 mod: Optional[str] = None, normalize_edges: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.cell_feature_channel = cell_feature_channel
        self.gene_feature_channel = gene_feature_channel or cell_feature_channel
        self.mod = mod
        self.normalize_edges = normalize_edges

    def __call__(self, data):
        feat = data.get_feature(return_type="default", mod=self.mod)
        num_cells, num_feats = feat.shape

        row, col = np.nonzero(feat)
        edata = np.array(feat[row, col])[:, None]
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

    def __init__(self, cell_feature_channel: str,  *,
                 mod: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.cell_feature_channel = cell_feature_channel
        self.mod = mod

    def __call__(self, data):
        feat = data.get_feature(channel_type=self.cell_feature_channel, return_type="sparse", mod=self.mod)
        g = dgl.bipartite_from_scipy(feat, utype='cell', etype='cell2feature', vtype='feature', eweight_name='weight')
        g.nodes['cell'].data['id'] = torch.arange(feat.shape[0]).long()
        g.nodes['gene'].data['id'] = torch.arange(feat.shape[1]).long()
        g = AddReverse(copy_edata=True, sym_new_etype=True)(g)
        if self.mod is None:
            data.data.uns['g'] = g
        else:
            data.data[self.mod].uns['g'] = g
        return data

# TODO:Probably move to model part
class CellFeatureBipartitePropagation(BaseTransform):
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 cell_init: str = None,
                 feature_init: str = 'id',
                 device: str = 'cuda',
                 layers: int = 3,
                 mod: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.mod = mod
        assert layers > 2, 'Less than two feature graph propagation layers is equivalent to original features.'
        self.layers = layers
        self.alpha = alpha
        self.beta = beta
        self.cell_init = cell_init
        self.feature_init = feature_init
        self.device = device

    def __call__(self, data):
        if self.mod is None:
            g = data.data.uns['g']
        else:
            g = data.data[self.mod].uns['g'].to(self.device)
        gconv = dglnn.HeteroGraphConv(
            {
                'cell2feature': dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
                'rev_cell2feature': dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
            }, aggregate='sum')

        if self.feature_init is None:
            feature_X = torch.zeros((g.nodes('feature').shape[0], g.srcdata[self.cell_init]['cell'].shape[1])).to(self.device)
        elif self.feature_init == 'id':
            feature_X = F.one_hot(g.srcdata['id']['feature']).float().to(self.device)
        else:
            raise NotImplementedError(f'Not implemented feature init feature {self.feature_init}.')

        if self.cell_init is None:
            cell_X = torch.zeros(g.nodes('cell').shape[0], feature_X.shape[1]).to(self.device)
        else:
            cell_X = g.srcdata[self.cell_init]['cell']

        h = {'feature': feature_X, 'cell': cell_X}
        hcell = []
        for i in range(self.layers):
            h1 = gconv(
                g, h, mod_kwargs={
                    'cell2feature': {
                        'edge_weight': g.edges['cell2feature'].data['weight']
                    },
                    'rev_cell2feature': {
                        'edge_weight': g.edges['rev_cell2feature'].data['weight']
                    }
                })
            # if verbose: print(i, 'cell', h['cell'].abs().mean(), h1['cell'].abs().mean())
            # if verbose: print(i, 'feature', h['feature'].abs().mean(), h1['feature'].abs().mean())

            h1['feature'] = (h1['feature'] -
                             h1['feature'].mean()) / (h1['feature'].std() if h1['feature'].mean() != 0 else 1)
            h1['cell'] = (h1['cell'] - h1['cell'].mean()) / (h1['cell'].std() if h1['cell'].mean() != 0 else 1)

            h = {
                'feature': h['feature'] * self.alpha + h1['feature'] * (1 - self.alpha),
                'cell': h['cell'] * self.beta + h1['cell'] * (1 - self.beta)
            }

            h['feature'] = (h['feature'] - h['feature'].mean()) / h['feature'].std()
            h['cell'] = (h['cell'] - h['cell'].mean()) / h['cell'].std()

            hcell.append(h['cell'])

        # if verbose: print(hcell[-1].abs().mean())

        # hcell = torch.cat(hcell[1:], dim=1)
        # if self.mod is None:
        #     data.data.obsm['prop'] = hcell[1:]
        # else:
        #     data.data[self.mod].obsm['prop'] = hcell[1:]

        if self.mod is None:
            data.data.uns['prop'] = hcell[1:]
        else:
            data.data[self.mod].uns['prop'] = hcell[1:]
        return data