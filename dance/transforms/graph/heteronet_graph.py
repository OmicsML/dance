from typing import Optional

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform


@register_preprocessor("graph", "cell")
class HeteronetGraph(BaseTransform):

    def __init__(self, knn_num: int = 5, distance_metrics: str = 'l2', random_state: int = 0,
                 channel: Optional[str] = None, channel_type: Optional[str] = "X", ignore_first: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.knn_num = knn_num
        self.distance_metrics = distance_metrics
        self.random_state = random_state
        self.channel = channel
        self.ignore_first = ignore_first
        self.channel_type = channel_type

    def build_graph(self, features_np, radius=None, knears=None, distance_metrics='l2'):
        """
        based on https://github.com/hannshu/st_datasets/blob/master/utils/preprocess.py
        """
        coor = pd.DataFrame(features_np)
        if (radius):
            nbrs = NearestNeighbors(radius=radius, metric=distance_metrics).fit(coor)
            _, indices = nbrs.radius_neighbors(coor, return_distance=True)
        else:
            nbrs = NearestNeighbors(n_neighbors=knears + 1, metric=distance_metrics).fit(coor)
            _, indices = nbrs.kneighbors(coor)

        edge_list = np.array([[i, j] for i, sublist in enumerate(indices) for j in sublist])
        return edge_list

    def __call__(self, data):
        """Builds a DGL graph from an AnnData object.

        Args:
            adata: The AnnData object containing features, labels, and splits.
            ref_adata_name: Name for the dataset (used if needed later).
            knn_num: Number of nearest neighbors for graph construction.
            distance_metrics: Distance metric for KNN.
            ignore_first: If True, sets label 0 to -1.

        Returns:
            dgl.DGLGraph: A DGL graph with node features ('feat'), labels ('label'),
                        and split masks ('train_mask', 'val_mask', 'test_mask',
                        'id_mask', 'ood_mask').

        """
        adata = data.data
        # 1. Extract Features
        features_np = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        features = torch.as_tensor(features_np, dtype=torch.float32)  # Ensure float32 common for features
        num_nodes = features.shape[0]

        # 2. Extract Labels
        # Assuming labels are one-hot encoded in obsm['cell_type']
        labels_np = np.argmax(adata.obsm['cell_type'].copy(), axis=1)
        labels = torch.as_tensor(labels_np, dtype=torch.long)

        batchs = adata.obs.get('batch_id', None)

        if self.ignore_first:
            labels[labels == 0] = -1  # Apply ignore_first logic

        # 3. Build Edges using the provided build_graph function
        # Note: DGL also has dgl.knn_graph, which could be an alternative
        edge_list_np = self.build_graph(features_np, knears=self.knn_num, distance_metrics=self.distance_metrics)
        if edge_list_np.shape[0] == 0:
            # Create an empty graph if no edges
            src = torch.tensor([], dtype=torch.long)
            dst = torch.tensor([], dtype=torch.long)
        else:
            # DGL expects source and destination node tensors
            edge_list_tensor = torch.tensor(edge_list_np.T, dtype=torch.long)
            src, dst = edge_list_tensor[0], edge_list_tensor[1]

        # 4. Create DGL Graph
        g = dgl.graph((src, dst), num_nodes=num_nodes)

        # 5. Add Node Features and Labels
        g.ndata['feat'] = features
        g.ndata['label'] = labels
        if batchs is not None:
            g.ndata['batch_id'] = torch.from_numpy(batchs.values.astype(int)).long()
        adata.uns[self.out] = g
