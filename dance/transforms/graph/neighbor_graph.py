import scanpy as sc

from dance.transforms.base import BaseTransform
from dance.typing import Optional


class NeighborGraph(BaseTransform):
    """Construct neighborhood graph of observations.

    This is a thin wrapper of the :func:`scanpy.pp.neighbors` class and uses the ``connectivities`` as the adjacency
    matrix. If you want full flexibility and support from the :func:`scanpy.pp.neighbors` method, please consider using
    the interface :class:`~dance.transforms.interface.AnnDataTransform`.

    Parameters
    ----------
    n_neighbors
        Number of neighbors.
    n_pcs
        Number of PCs to use.
    knn
        If ``True``, then use a hard threshold to restrict the number of neighbors to ``n_neighbors``.
    random_state
        Random seed.
    method
        Method for computing the connectivities.
    metric
        Distance metric.
    channel
        Name of the PC channel.

    """

    _DISPLAY_ATTRS = ("n_neighbors", "n_pcs", "knn", "random_state", "method", "metric")

    def __init__(self, n_neighbors: int = 15, *, n_pcs: Optional[int] = None, knn: bool = True, random_state: int = 0,
                 method: Optional[str] = "umap", metric: str = "euclidean", channel: Optional[str] = "CellPCA",
                 **kwargs):
        super().__init__(**kwargs)

        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.knn = knn
        self.random_state = random_state
        self.method = method
        self.metric = metric
        self.channel = channel

    def __call__(self, data):
        self.logger.info("Start computing the kNN connectivity adjacency matrix")
        adj = sc.pp.neighbors(data.data, copy=True, use_rep=self.channel, n_neighbors=self.n_neighbors,
                              n_pcs=self.n_pcs, knn=self.knn, random_state=self.random_state, method=self.method,
                              metric=self.metric).obsp["connectivities"]
        data.data.obsp[self.out] = adj

        return data
