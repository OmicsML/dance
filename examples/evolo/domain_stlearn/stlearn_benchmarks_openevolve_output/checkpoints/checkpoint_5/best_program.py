import argparse
import os
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
import torch
from tqdm import tqdm, trange
import wandb

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.stlearn import StLouvain
from dance.registry import register_preprocessor
from dance.utils import set_seed, sub_data
import scanpy as sc

from dance.transforms import AnnDataTransform, BaseTransform, Compose, SetConfig
from dance.typing import LogLevel, Optional, Sequence, Union
from dance.utils.matrix import normalize
from dance.utils.metrics import calculate_unified_scores, resolve_score_func
from dance.utils.wrappers import add_mod_and_transform

MODES = ["louvain", "kmeans"]
# EVOLVE-BLOCK-START

@register_preprocessor("feature", "spatial",overwrite=True)
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
                 batch_size: int = 64, **kwargs):
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
        self.model.eval()

    def _crop_and_process_batch(self, image, coords):
        """Crop and process multiple patches in batch"""
        crops = []
        for x, y in coords:
            cs = self.crop_size
            ts = self.target_size
            
            # Handle boundary conditions
            x_start = max(0, int(x - cs))
            x_end = int(x + cs)
            y_start = max(0, int(y - cs))
            y_end = int(y + cs)
            
            # Extract patch
            patch = image[y_start:y_end, x_start:x_end, :]
            
            # Resize patch to target size
            if patch.shape[0] > 0 and patch.shape[1] > 0:  # Check if valid patch
                patch = cv2.resize(patch, (ts, ts))
            else:
                # Create black patch if crop fails
                patch = np.zeros((ts, ts, 3), dtype=np.uint8)
                
            # Normalize
            patch = (patch - self.mean) / self.std
            patch = patch.transpose((2, 0, 1))
            crops.append(patch)
        
        return torch.FloatTensor(np.stack(crops))

    def __call__(self, data):
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        image = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])

        # Process in batches for computational efficiency
        features = []
        n_samples = len(xy_pixel)
        
        for i in trange(0, n_samples, self.batch_size, desc="Extracting morphology features", 
                       bar_format="{l_bar}{bar} [ time left: {remaining} ]"):
            batch_coords = xy_pixel[i:i+self.batch_size]
            batch_images = self._crop_and_process_batch(image, batch_coords).to(self.device)
            
            with torch.no_grad():  # Disable gradient computation for inference
                batch_features = self.model(batch_images).detach().cpu().numpy()
                features.extend(batch_features)
        
        morth_feat = np.array(features)
        if self.n_components > 0:
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            morth_feat = pca.fit_transform(morth_feat)

        data.data.obsm[self.out] = morth_feat


@register_preprocessor("feature", "spatial",overwrite=True)
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

    def __init__(self, n_neighbors: int = 5, n_components: int = 50, random_state: int = 0, *,
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

        # Vectorized operation using sparse matrix operations to efficiently calculate neighbor imputation
        num_samples, num_genes = x.shape
        
        # Find top-k neighbors for each sample
        top_k_indices = np.argpartition(adj, -self.n_neighbors, axis=1)[:, -self.n_neighbors:]
        top_k_values = np.take_along_axis(adj, top_k_indices, axis=1)
        
        # Normalize weights to sum to 1 for each sample
        sums = top_k_values.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1  # Prevent division by zero
        normalized_weights = top_k_values / sums
        
        # Create a result array for imputed values
        imputed = np.zeros_like(x)
        
        # Efficiently compute weighted averages using broadcasting
        for i in range(num_samples):
            neighbor_indices = top_k_indices[i]
            neighbor_weights = normalized_weights[i, :, np.newaxis]  # Add dimension for broadcasting
            neighbor_values = x[neighbor_indices]
            imputed[i] = np.sum(neighbor_weights * neighbor_values, axis=0)

        sme_feat = (x + imputed) / 2
        if self.n_components > 0:
            sme_feat = normalize(sme_feat, mode="standardize", axis=0)
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            sme_feat = pca.fit_transform(sme_feat)

        data.data.obsm[self.out] = sme_feat
        

@register_preprocessor("feature", "cell",overwrite=True)
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

@register_preprocessor("graph", "spatial",overwrite=True)
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

        # Calculate pairwise distances
        pdist = pairwise_distances(xy_pixel, metric="euclidean")
        
        # Create adjacency matrices for each modality
        adj_p = np.where(pdist >= self.radius * unit, 0, 1)
        adj_m = (1 - pairwise_distances(morph_feat, metric="cosine")).clip(0)
        adj_g = 1 - pairwise_distances(gene_feat, metric="correlation")
        
        # Use weighted combination instead of multiplicative to prevent over-sparsification
        alpha, beta, gamma = 0.4, 0.3, 0.3  # Adjusted weights
        adj = alpha * adj_p + beta * adj_m + gamma * adj_g
        adj = np.clip(adj, 0, 1)  # Ensure values are in [0,1]

        data.data.obsp[self.out] = adj
        
@register_preprocessor("graph", "cell",overwrite=True)
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
     
          
def get_preprocessing_pipeline(morph_feat_dim: int = 50, sme_feat_dim: int = 50, pca_feat_dim: int = 50,
                            nbrs_pcs: int = 50, n_neighbors: int = 15, device: str = "cpu",
                            log_level: LogLevel = "INFO", crop_size=15, target_size=224):
    return Compose(
        AnnDataTransform(sc.pp.filter_genes, min_cells=1),
        AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
        AnnDataTransform(sc.pp.log1p),
        MorphologyFeatureCNN(n_components=morph_feat_dim, device=device, crop_size=crop_size,
                                target_size=target_size, batch_size=64),
        CellPCA(n_components=pca_feat_dim),
        SMEGraph(),
        SMEFeature(n_components=sme_feat_dim, n_neighbors=5),
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
    parser.add_argument("--obs_nums",type=int,default=10000)
    args = parser.parse_args()
    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(args.seed)

        # Initialize model and get model specific preprocessing pipeline
        if args.mode == "kmeans":
            raise NotImplementedError("--------")
        elif args.mode == "louvain":
            model = StLouvain(resolution=0.6,random_state=seed)
        else:
            raise ValueError(f"Unknown mode {args.mode!r}, available options are {MODES}")
        # preprocessing_pipeline = model.preprocessing_pipeline(device=args.device)

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number, data_dir=args.data_dir,
                                        sample_file=args.sample_file)
        preprocessing_pipeline = get_preprocessing_pipeline(device=args.device)
        data = dataloader.load_data(transform=None,cache=args.cache)
        sub_data(data.data,args.obs_nums)
        preprocessing_pipeline(data)
        # Prepare preprocessing pipeline and apply it to data
        x, y = data.get_data(return_type="default")
        score = model.fit_score(x, y.values.ravel())
        pred=model.predict()
        silhouette_score = resolve_score_func("silhouette")
        calinski_harabasz_score = resolve_score_func("calinski_harabasz")
        davies_bouldin_score = resolve_score_func("davies_bouldin")
        inner_scores.append(calculate_unified_scores({
            "silhouette": silhouette_score(x.toarray(), pred),
            "calinski_harabasz": calinski_harabasz_score(x.toarray(), pred),
            "davies_bouldin": davies_bouldin_score(x.toarray(), pred)
        }))
        scores.append(score)
        print(f"ARI: {score:.4f}")
    print(f"STAGATE {args.sample_number}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
        

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
