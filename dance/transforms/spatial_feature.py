import logging

import cv2
import numpy as np
import pandas as pd
import patsy
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

from dance.data.base import Data
from dance.transforms.base import BaseTransform
from dance.typing import Optional, Sequence
from dance.utils.matrix import normalize


class MorphologyFeature(BaseTransform):

    _DISPLAY_ATTRS = ("model_name", "n_components", "crop_size", "target_size")
    _MODELS = ("resnet50", "inception_v3", "xception", "vgg16")

    def __init__(self, *, model_name: str = "resnet50", n_components: int = 50, random_state: int = 0,
                 crop_size: int = 20, target_size: int = 299, device: str = "cpu",
                 channels: Sequence[str] = ("spatial_pixel", "image"), channel_types: Sequence[str] = ("obsm", "uns"),
                 **kwargs):
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

        self.mean = np.array([0.406, 0.485, 0.456])
        self.std = np.array([0.225, 0.229, 0.224])

        if self.model_name not in self._MODELS:
            raise ValueError(f"Unsupported model {self.model_name!r}, available options are: {self._MODELS}")
        self.model = getattr(tv.models, self.model_name)(pretrained=True)
        self.model.fc = torch.nn.Sequential()
        self.model = self.model.to(self.device)

    def _crop_and_process(self, image, x, y):
        cs = self.crop_size
        ts = self.target_size

        img = image[int(x - cs):int(x + cs), int(y - cs):int(y + cs), :]
        img = cv2.resize(img, (ts, ts))
        img = (img - self.mean) / self.std
        img = img.transpose((2, 0, 1))
        img = torch.FloatTensor(img).unsqueeze(0)
        return img

    def __call__(self, data):
        xy_pixel = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        image = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])

        # TODO: improve computational efficiency by processing images in batch.
        features = []
        for x, y in tqdm(xy_pixel, desc="Extracting feature", bar_format="{l_bar}{bar} [ time left: {remaining} ]"):
            img = self._crop_and_process(image, x, y).to(self.device)
            feature = self.model(img).view(-1).detach().cpu().numpy()
            features.append(feature)

        morth_feat = np.array(features)
        if self.n_components > 0:
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            morth_feat = pca.fit_transform(morth_feat)

        data.data.obsm[self.out] = morth_feat


class SMEFeature(BaseTransform):

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

        imputed = []
        num_samples, num_genes = x.shape
        for i in trange(num_samples, desc="Adjusting data", bar_format="{l_bar}{bar} [ time left: {remaining} ]"):
            weights = adj[i]
            nbrs_idx = weights.argsort()[-self.n_neighbors:]
            nbrs_weights = weights[nbrs_idx]

            if nbrs_weights.sum() > 0:
                nbrs_weights_scaled = (nbrs_weights / nbrs_weights.sum())
                aggregated = (nbrs_weights_scaled[:, None] * x[nbrs_idx]).sum(0)
            else:
                aggregated = x[i]

            imputed.append(aggregated)

        sme_feat = (x + np.array(imputed)) / 2
        if self.n_components > 0:
            sme_feat = normalize(sme_feat, mode="standardize", axis=0)
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            sme_feat = pca.fit_transform(sme_feat)

        data.data.obsm[self.out] = sme_feat


class SpatialIDEFeature(BaseTransform):
    """Spatial IDE feature.

    The SpatialDE model is based on the assumption of normally distributed residual noise and independent observations
    across cells. There are two normalization steps:

        1. Variance-stabilizing transformation for negative-binomial-distributed data (Anscombe's transformation).
        2. Regress log total count values out from the Anscombe-transformed expression values.

    Reference
    ---------
    https://www.nature.com/articles/nmeth.4636#Sec2

    """

    def __init__(self, channels: Sequence[Optional[str]] = (None, "spatial"),
                 channel_types: Sequence[Optional[str]] = (None, "obsm"), **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.channel_types = channel_types

    def regress_out(self, sample_info, expression_matrix, covariate_formula, design_formula='1', rcond=-1):
        """Implementation of limma's removeBatchEffect function."""
        # Ensure intercept is not part of covariates
        covariate_formula += ' - 1'

        covariate_matrix = patsy.dmatrix(covariate_formula, sample_info)
        design_matrix = patsy.dmatrix(design_formula, sample_info)

        design_batch = np.hstack((design_matrix, covariate_matrix))

        coefficients, res, rank, s = np.linalg.lstsq(design_batch, expression_matrix.T, rcond=rcond)
        beta = coefficients[design_matrix.shape[1]:]
        regressed = expression_matrix - covariate_matrix.dot(beta).T

        return regressed

    def stabilize(self, expression_matrix):
        """Use Anscombes approximation to variance stabilize Negative Binomial data.

        See https://f1000research.com/posters/4-1041 for motivation.

        Assumes columns are samples, and rows are genes

        """
        from scipy import optimize
        phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu**2, expression_matrix.mean(1),
                                        expression_matrix.var(1))

        return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

    def __call__(self, data):
        counts = data.get_feature(return_type="numpy", channel=self.channels[0], channel_type=self.channel_types[0])
        xy = data.get_feature(return_type="numpy", channel=self.channels[1], channel_type=self.channel_types[1])
        norm_expr = self.stabilize(counts.T).T
        sample_info = pd.DataFrame(xy, columns=['x', 'y'])
        sample_info['total_counts'] = np.sum(counts, axis=1)
        resid_expr = self.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T
        data.data.obsm[self.out] = resid_expr


class TangramFeature(BaseTransform):
    """Tangram spatial features.

    First, compute the cell density inside each voxel. Then, the cell density distributions are compared using
    Kullback-Leibler (KL) divergence, whereas gene expression is assessed via cosine similarity.

    Reference
    ---------
    https://www.nature.com/articles/s41592-021-01264-7

    """

    def __init__(self, density_mode: str = "uniform", channel: Optional[str] = None, channel_type: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.channel_type = channel_type
        self.density_mode = density_mode

    def __call__(self, data: Data) -> Data:
        x = data.get_feature(return_type="default", channel=self.channel, channel_type=self.channel_type)
        if self.density_mode == "uniform":
            logging.info("Calculating uniform based density prior.")
            density = np.ones(x.shape[0]) / x.shape[0]
        elif self.density_mode == "rna_count":
            # Calculate rna_count_based density prior as % of rna molecule count
            logging.info("Calculating rna count based density prior.")
            rna_count_per_spot = np.array(x.sum(axis=1)).squeeze()
            density = rna_count_per_spot / np.sum(rna_count_per_spot)
        else:
            raise ValueError(f"Unknwon density mode {self.density_mode!r}, "
                             "supported options are: 'uniform', 'rna_count'")
        data.data.obs[self.out] = density
        return data
