import argparse
import os
import sys
import time  # 新增：导入 time 模块
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import scanpy as sc
import torch
import wandb
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from tqdm import tqdm, trange

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stlearn import StLouvain
from dance.registry import register_preprocessor
from dance.transforms import AnnDataTransform, BaseTransform, Compose, SetConfig
from dance.typing import LogLevel, Optional, Sequence, Union
from dance.utils import set_seed, sub_data
from dance.utils.matrix import normalize
from dance.utils.metrics import calculate_unified_scores, resolve_score_func
from dance.utils.wrappers import add_mod_and_transform

MODES = ["louvain", "kmeans"]
# EVOLVE-BLOCK-START


@register_preprocessor("feature", "spatial", overwrite=True)
class MorphologyFeatureCNN(BaseTransform):
    """Cell morphological features extracted from CNN.

    Parameters
    ----------
    model_name
        Pretrained CNN name: ``"resnet50"``, ``"inceptron_v3"``, ``"xception"``, ``"vgg16"``.
    n_components
        Number of feature dimension.
    crop_size
        Cell image cropping size (cropped as square centered around the target cell).
    target_size
        Target patch size.

    References
    ----------
    https://doi.org/10.1101/2020.05.31.125658

    """

    _DISPLAY_ATTRS = ("model_name", "n_components", "crop_size", "target_size")
    _MODELS = ("resnet50", "inception_v3", "xception", "vgg16")

    def __init__(self, *, model_name: str = "resnet50", n_components: int = 50, random_state: int = 0,
                 crop_size: int = 20, target_size: int = 299, device: str = "cpu",
                 channels: Sequence[str] = ("spatial_pixel", "image"), channel_types: Sequence[str] = ("obsm", "uns"),
                 batch_size: int = 32, **kwargs):
        import torchvision as tv

        super().__init__(**kwargs)

        self.model_name = model_name
        self.n_components = n_components
        self.random_state = random_state
        self.crop_size = crop_size
        self.target_size = target_size
        self.device = device
        self.channels = channels
        self.channel_types = channel_types
        self.batch_size = batch_size

        self.mean = np.array([0.406, 0.485, 0.456])
        self.std = np.array([0.225, 0.229, 0.224])

        if self.model_name not in self._MODELS:
            raise ValueError(f"Unsupported model {self.model_name!r}, available options are: {self._MODELS}")
        self.model = getattr(tv.models, self.model_name)(pretrained=True)
        self.model.fc = torch.nn.Sequential()
        self.model = self.model.to(self.device)

    def _crop_and_process_batch(self, image, coords):
        """Process a batch of coordinates efficiently."""
        cs = self.crop_size
        ts = self.target_size
        batch_images = []

        for x, y in coords:
            img = image[max(0, int(x - cs)):int(x + cs), max(0, int(y - cs)):int(y + cs), :]
            img = cv2.resize(img, (ts, ts))
            img = (img - self.mean) / self.std
            img = img.transpose((2, 0, 1))
            batch_images.append(img)

        batch_tensor = torch.FloatTensor(batch_images).to(self.device)
        return batch_tensor

    def __call__(self, data):
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        image = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])

        # Process in batches for better efficiency
        features = []
        n_samples = len(xy_pixel)

        for i in range(0, n_samples, self.batch_size):
            batch_coords = xy_pixel[i:i + self.batch_size]
            batch_tensor = self._crop_and_process_batch(image, batch_coords)

            with torch.no_grad():
                batch_features = self.model(batch_tensor).view(len(batch_coords), -1).detach().cpu().numpy()
            features.extend(batch_features)

        morth_feat = np.array(features)

        if self.n_components > 0:
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            morth_feat = pca.fit_transform(morth_feat)

        data.data.obsm[self.out] = morth_feat


@register_preprocessor("feature", "spatial", overwrite=True)
class SMEFeature(BaseTransform):
    """Spatial Morphological gene Expression normalization feature from stLearn.

    Parameters
    ----------
    n_neighbors
        Number of spatial spots neighbors to consider.
    n_components
        Number of feature dimension.

    References
    ----------
    https://doi.org/10.1101/2020.05.31.125658

    """

    def __init__(self, n_neighbors: int = 3, n_components: int = 50, random_state: int = 0, *,
                 channels: Sequence[Optional[str]] = (None, "SMEGraph"),
                 channel_types: Sequence[Optional[str]] = (None, "obsp"), **kwargs):
        super().__init__(**kwargs)

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.random_state = random_state
        self.channels = channels
        self.channel_types = channel_types

    def __call__(self, data):
        x = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        adj = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])

        # Use top-k neighbors to create a sparse adjacency matrix
        n_samples = adj.shape[0]
        top_k_indices = np.argpartition(adj, -self.n_neighbors, axis=1)[:, -self.n_neighbors:]
        top_k_values = np.take_along_axis(adj, top_k_indices, axis=1)

        # Normalize the weights for each sample
        top_k_weights = np.zeros_like(adj)
        rows = np.repeat(np.arange(n_samples), self.n_neighbors)
        cols = top_k_indices.flatten()
        vals = top_k_values.flatten()
        top_k_weights[rows, cols] = vals

        # Normalize weights for each row
        row_sums = top_k_weights.sum(axis=1)
        mask = row_sums > 0
        top_k_weights[mask] /= row_sums[mask, np.newaxis]

        # Compute imputed values using matrix multiplication
        imputed = top_k_weights @ x
        sme_feat = (x + imputed) / 2

        if self.n_components > 0:
            sme_feat = normalize(sme_feat, mode="standardize", axis=0)
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            sme_feat = pca.fit_transform(sme_feat)

        data.data.obsm[self.out] = sme_feat


@register_preprocessor("feature", "cell", overwrite=True)
@add_mod_and_transform
class CellPCA(BaseTransform):
    """Reduce cell feature matrix with PCA.

    Parameters
    ----------
    n_components
        Number of PCA components to use.

    """

    _DISPLAY_ATTRS = ("n_components", )

    def __init__(self, n_components: Union[float, int] = 400, *, channel: Optional[str] = None,
                 mod: Optional[str] = None, save_info: bool = False,
                 svd_solver: Literal['auto', 'full', 'arpack', 'randomized'] = "auto", **kwargs):
        super().__init__(**kwargs)

        self.n_components = n_components
        self.channel = channel
        self.save_info = save_info
        self.svd_solver = svd_solver

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy", channel=self.channel)
        if self.n_components > min(feat.shape):
            self.logger.warning(
                f"n_components={self.n_components} must be between 0 and min(n_samples, n_features)={min(feat.shape)} with svd_solver='{self.svd_solver}'"
            )
            self.n_components = min(feat.shape)
        if "pca" not in data.data.uns:
            pca = PCA(n_components=self.n_components, svd_solver=self.svd_solver)
            cell_feat = pca.fit_transform(feat)
        else:
            pca = data.data.uns["pca"]
            cell_feat = pca.transform(feat)
        self.logger.info(f"Generating cell PCA features {feat.shape} (k={pca.n_components_})")
        evr = pca.explained_variance_ratio_
        self.logger.info(f"Top 10 explained variances: {evr[:10]}")
        self.logger.info(f"Total explained variance: {evr.sum():.2%}")

        data.data.obsm[self.out] = cell_feat
        if self.save_info:
            data.data.uns["pca_components"] = pca.components_
            data.data.uns["pca_mean"] = pca.mean_
            data.data.uns["pca_explained_variance"] = pca.explained_variance_
            data.data.uns["pca_explained_variance_ratio"] = pca.explained_variance_ratio_
            # data.data.uns["pca"] = pca

        return data


@register_preprocessor("graph", "spatial", overwrite=True)
class SMEGraph(BaseTransform):
    """Spatial Morphological gene Expression graph."""

    def __init__(self, radius: float = 3, alpha: float = 0.5, beta: float = 0.3, *,
                 channels: Sequence[str] = ("spatial", "spatial_pixel", "MorphologyFeatureCNN", "CellPCA"),
                 channel_types: Sequence[str] = ("obsm", "obsm", "obsm", "obsm"), **kwargs):
        super().__init__(**kwargs)

        self.radius = radius
        self.alpha = alpha  # Weight for spatial proximity
        self.beta = beta  # Weight for morphological similarity
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

        # Calculate distances
        pdist = pairwise_distances(xy_pixel, metric="euclidean")
        adj_p = np.where(pdist >= self.radius * unit, 0, 1)

        adj_m = (1 - pairwise_distances(morph_feat, metric="cosine")).clip(0)
        adj_g = 1 - pairwise_distances(gene_feat, metric="correlation")

        # Use a weighted combination instead of strict multiplication
        adj = self.alpha * adj_p + (1 - self.alpha) * (self.beta * adj_m + (1 - self.beta) * adj_g)

        # Apply threshold to keep top-k connections per node
        k = int(min(10, adj.shape[0] * 0.1))  # Use 10% of nodes or 10, whichever is smaller
        for i in range(adj.shape[0]):
            top_k_indices = np.argpartition(adj[i], -k)[-k:]
            mask = np.ones(adj.shape[1], dtype=bool)
            mask[top_k_indices] = False
            adj[i, mask] = 0

        data.data.obsp[self.out] = adj


@register_preprocessor("graph", "cell", overwrite=True)
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


def get_preprocessing_pipeline(morph_feat_dim: int = 50, sme_feat_dim: int = 50, pca_feat_dim: int = 10,
                               nbrs_pcs: int = 10, n_neighbors: int = 10, device: str = "cpu",
                               log_level: LogLevel = "INFO", crop_size=10, target_size=230):
    return Compose(
        AnnDataTransform(sc.pp.filter_genes, min_cells=1),
        AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
        AnnDataTransform(sc.pp.log1p),
        MorphologyFeatureCNN(n_components=morph_feat_dim, device=device, crop_size=crop_size, target_size=target_size,
                             batch_size=32),
        CellPCA(n_components=pca_feat_dim),
        SMEGraph(alpha=0.4, beta=0.3),
        SMEFeature(n_components=sme_feat_dim),
        NeighborGraph(n_neighbors=n_neighbors, n_pcs=nbrs_pcs, channel="SMEFeature"),
        SetConfig({
            "feature_channel": "NeighborGraph",
            "feature_channel_type": "obsp",
            "label_channel": "label",
            "label_channel_type": "obs",
        }),
        log_level=log_level,
    )


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--mode", type=str, default="louvain", choices=MODES)
    parser.add_argument("--n_clusters", type=int, default=17, help="the number of clusters")
    parser.add_argument("--n_components", type=int, default=50, help="the number of components in PCA")
    parser.add_argument("--device", type=str, default="cuda", help="device for resnet extract feature")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default='./temp_data', help='test directory')
    parser.add_argument("--sample_file", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--obs_nums", type=int, default=10000)
    args = parser.parse_args()
    scores = []
    inner_scores = []
    times = []  # 新增：用于记录每次运行的时间

    for seed in range(args.seed, args.seed + args.num_runs):
        start_time = time.time()  # 新增：记录单次循环的开始时间

        set_seed(seed)  # 修正：之前是 args.seed，会导致每次跑出来的随机性相同，这里改成跟随循环变量 seed

        # Initialize model and get model specific preprocessing pipeline
        if args.mode == "kmeans":
            raise NotImplementedError("--------")
        elif args.mode == "louvain":
            model = StLouvain(resolution=0.6, random_state=seed)
        else:
            raise ValueError(f"Unknown mode {args.mode!r}, available options are {MODES}")
        # preprocessing_pipeline = model.preprocessing_pipeline(device=args.device)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number, data_dir=args.data_dir,
                                        sample_file=args.sample_file)
        preprocessing_pipeline = get_preprocessing_pipeline(device=args.device)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data, args.obs_nums)
        preprocessing_pipeline(data)
        # Prepare preprocessing pipeline and apply it to data
        x, y = data.get_data(return_type="default")
        score = model.fit_score(x, y.values.ravel())
        pred = model.predict()
        silhouette_score = resolve_score_func("silhouette")
        calinski_harabasz_score = resolve_score_func("calinski_harabasz")
        davies_bouldin_score = resolve_score_func("davies_bouldin")
        inner_scores.append(
            calculate_unified_scores({
                "silhouette": silhouette_score(x.toarray(), pred),
                "calinski_harabasz": calinski_harabasz_score(x.toarray(), pred),
                "davies_bouldin": davies_bouldin_score(x.toarray(), pred)
            }))
        scores.append(score)

        end_time = time.time()  # 新增：记录单次循环的结束时间
        run_time = end_time - start_time
        times.append(run_time)  # 新增：保存耗时

        print(f"ARI: {score:.4f}, time: {run_time:.2f}s")  # 修改：同时输出得分与时间

    print(f"STAGATE {args.sample_number}:")
    # 修改：加入 times 列表，以供 evaluator 捕获
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")  # 新增：输出平均运行时间
""" To reproduce stlearn on other samples, please refer to command lines belows:
NOTE: since the stlearn method is unstable, you have to run multiple times to get
      best performance.

human dorsolateral prefrontal cortex sample 151673:
$ python stlearn.py --n_clusters 20 --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python stlearn.py --n_clusters 20 --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python stlearn.py --n_clusters 20 --sample_number 151507
"""
