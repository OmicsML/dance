import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.typing import Sequence
from dance.utils.matrix import pairwise_distance


@register_preprocessor("graph", "spatial")
class CalSpatialNet(BaseTransform):
    """Construct the spatial neighbor networks.

    Parameters
    ----------
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    spatial_uns
        Key for storing spatial networks in adata.uns. Default is "Spatial_Net".

    """

    _DISPLAY_ATTRS = ("rad_cutoff", "k_cutoff", "model", "spatial_uns")

    def __init__(self, rad_cutoff=None, k_cutoff=None, model='Radius', spatial_uns="Spatial_Net", out=None,
                 log_level="WARNING"):
        super().__init__(out=out, log_level=log_level)
        self.rad_cutoff = rad_cutoff
        self.k_cutoff = k_cutoff
        self.model = model
        self.spatial_uns = spatial_uns

    def __call__(self, data):
        """Construct spatial neighbor networks."""
        adata = data.data

        print('------Calculating spatial graph...')
        assert (self.model in ['Radius', 'KNN'])
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']

        if self.model == 'Radius':
            nbrs = sklearn.neighbors.NearestNeighbors(radius=self.rad_cutoff).fit(coor)
            distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

        if self.model == 'KNN':
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k_cutoff + 1).fit(coor)
            distances, indices = nbrs.kneighbors(coor)
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

        Spatial_Net = KNN_df.copy()
        Spatial_Net = Spatial_Net.loc[
            Spatial_Net['Distance'] > 0,
        ]
        id_cell_trans = dict(zip(
            range(coor.shape[0]),
            np.array(coor.index),
        ))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

        adata.uns[self.spatial_uns] = Spatial_Net
        return data


@register_preprocessor("graph", "spatial")
class SpaGCNGraph(BaseTransform):

    _DISPLAY_ATTRS = ("alpha", "beta")

    def __init__(self, alpha, beta, *, channels: Sequence[str] = ("spatial", "spatial_pixel", "image"),
                 channel_types: Sequence[str] = ("obsm", "obsm", "uns"), **kwargs):
        """Initialize SpaGCNGraph.

        Parameters
        ----------
        alpha
            Controls the color scale.
        beta
            Controls the range of the neighborhood when calculating grey values for one spot.

        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.channels = channels
        self.channel_types = channel_types

    def __call__(self, data):
        xy = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])
        img = data.get_feature(return_type="numpy", channel=self.channels[2], channel_type=self.channel_types[2])
        self.logger.info("Start calculating the adjacency matrix using the histology image")
        g = np.zeros((xy.shape[0], 3))
        beta_half = round(self.beta / 2)
        x_lim, y_lim = img.shape[:2]
        for i, (x_pixel, y_pixel) in enumerate(xy_pixel):
            top = max(0, x_pixel - beta_half)
            left = max(0, y_pixel - beta_half)
            bottom = min(x_lim, x_pixel + beta_half + 1)
            right = min(y_lim, y_pixel + beta_half + 1)
            local_view = img[top:bottom, left:right]
            g[i] = np.mean(local_view, axis=(0, 1))
        g_var = g.var(0)
        self.logger.info(f"Variances of c0, c1, c2 = {g_var}")

        z = (g * g_var).sum(1, keepdims=True) / g_var.sum()
        z = (z - z.mean()) / z.std()
        z *= xy.std(0).max() * self.alpha

        xyz = np.hstack((xy, z)).astype(np.float32)
        self.logger.info(f"Varirances of x, y, z = {xyz.var(0)}")
        data.data.obsp[self.out] = pairwise_distance(xyz, dist_func_id=0)

        return data


@register_preprocessor("graph", "spatial")
class SpaGCNGraph2D(BaseTransform):

    def __init__(self, *, channel: str = "spatial_pixel", **kwargs):
        super().__init__(**kwargs)

        self.channel = channel

    def __call__(self, data):
        x = data.get_feature(channel=self.channel, channel_type="obsm", return_type="numpy")
        data.data.obsp[self.out] = pairwise_distance(x.astype(np.float32), dist_func_id=0)
        return data


@register_preprocessor("graph", "spatial")
class SMEGraph(BaseTransform):
    """Spatial Morphological gene Expression graph."""

    def __init__(self, radius: float = 3, *,
                 channels: Sequence[str] = ("spatial", "spatial_pixel", "MorphologyFeatureCNN", "CellPCA"),
                 channel_types: Sequence[str] = ("obsm", "obsm", "obsm", "obsm"), **kwargs):
        super().__init__(**kwargs)

        self.radius = radius
        self.channels = channels
        self.channel_types = channel_types

    def __call__(self, data):
        xy = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])
        morph_feat = data.get_feature(return_type="numpy", channel=self.channels[2], channel_type=self.channel_types[2])
        gene_feat = data.get_feature(return_type="numpy", channel=self.channels[3], channel_type=self.channel_types[3])

        reg_x = LinearRegression().fit(xy[:, 0:1], xy_pixel[:, 0:1])
        reg_y = LinearRegression().fit(xy[:, 1:2], xy_pixel[:, 1:2])
        unit = np.sqrt(reg_x.coef_**2 + reg_y.coef_**2)

        # TODO: only captures topk, which are the ones that will be used by SMEFeature.
        pdist = pairwise_distances(xy_pixel, metric="euclidean")
        adj_p = np.where(pdist >= self.radius * unit, 0, 1)
        adj_m = (1 - pairwise_distances(morph_feat, metric="cosine")).clip(0)
        adj_g = 1 - pairwise_distances(gene_feat, metric="correlation")
        adj = adj_p * adj_m * adj_g

        data.data.obsp[self.out] = adj


@register_preprocessor("graph", "spatial")
class StagateGraph(BaseTransform):
    """STAGATE spatial graph.

    Parameters
    ----------
    model_name
        Type of graph to construct. Currently support ``radius`` and ``knn``. See
        :class:`~sklearn.neighbors.NearestNeighbors` for more info.
    radius
        Radius parameter for ``radius_neighbors_graph``.
    n_neighbors
        Number of neighbors for ``kneighbors_graph``.

    """

    _MODELS = ("radius", "knn")
    _DISPLAY_ATTRS = ("model_name", "radius", "n_neighbors")

    def __init__(self, model_name: str = "radius", *, radius: float = 1, n_neighbors: int = 5,
                 channel: str = "spatial_pixel", channel_type: str = "obsm", **kwargs):
        super().__init__(**kwargs)

        if not isinstance(model_name, str) or (model_name.lower() not in self._MODELS):
            raise ValueError(f"Unknown model {model_name!r}, available options are {self._MODELS}")
        self.model_name = model_name
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.channel = channel
        self.channel_type = channel_type

    def __call__(self, data):
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)

        if self.model_name.lower() == "radius":
            adj = NearestNeighbors(radius=self.radius).fit(xy_pixel).radius_neighbors_graph(xy_pixel)
        elif self.model_name.lower() == "knn":
            adj = NearestNeighbors(n_neighbors=self.n_neighbors).fit(xy_pixel).kneighbors_graph(xy_pixel)

        data.data.obsp[self.out] = adj
