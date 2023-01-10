import cv2
import numpy as np
import torch
import torchvision as tv
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

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

        img = image[x - cs:x + cs, y - cs:y + cs, :]
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
