import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dance.transforms.base import BaseTransform
from dance.typing import Optional
from dance.utils.matrix import normalize


class WeightedFeaturePCA(BaseTransform):
    """Compute the weighted gene PCA as cell features.

    Given a gene expression matrix of dimension (cell x gene), the gene PCA is first compured. Then, the representation
    of each cell is computed by taking the weighted sum of the gene PCAs based on that cell's gene expression values.

    Parameters
    ----------
    n_components
        Number of PCs to use.
    split_name
        Which split to use to compute the gene PCA. If not set, use all data.

    """

    _DISPLAY_ATTRS = ("n_components", "split_name", "feat_norm_mode", "feat_norm_axis")

    def __init__(self, n_components: int = 400, split_name: Optional[str] = None, feat_norm_mode: Optional[str] = None,
                 feat_norm_axis: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.n_components = n_components
        self.split_name = split_name
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis

    def __call__(self, data):
        feat = data.get_x(self.split_name)  # cell x genes
        if self.feat_norm_mode is not None:
            self.logger.info(f"Normalizing feature before PCA decomposition with mode={self.feat_norm_mode} "
                             f"and axis={self.feat_norm_axis}")
            feat = normalize(feat, mode=self.feat_norm_mode, axis=self.feat_norm_axis)
        gene_pca = PCA(n_components=self.n_components)

        self.logger.info(f"Start decomposing {self.split_name} features {feat.shape} (k={self.n_components})")
        gene_feat = gene_pca.fit_transform(feat.T)  # decompose into gene features
        self.logger.info(f"Total explained variance: {gene_pca.explained_variance_ratio_.sum():.2%}")

        x = data.get_x()
        cell_feat = normalize(x, mode="normalize", axis=1) @ gene_feat
        data.data.obsm[self.out] = cell_feat.astype(np.float32)
        data.data.varm[self.out] = gene_feat.astype(np.float32)

        return data


class CellPCA(BaseTransform):

    _DISPLAY_ATTRS = ("n_components", )

    def __init__(self, n_components: int = 400, *, channel: Optional[str] = None, mod: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.n_components = n_components
        self.channel = channel
        self.mod = mod

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy", channel=self.channel, mod=self.mod)
        pca = PCA(n_components=self.n_components)

        self.logger.info(f"Start generating cell PCA features {feat.shape} (k={self.n_components})")
        cell_feat = pca.fit_transform(feat)
        evr = pca.explained_variance_ratio_
        self.logger.info(f"Top 10 explained variances: {evr[:10]}")
        self.logger.info(f"Total explained variance: {evr.sum():.2%}")

        data.data.obsm[self.out] = cell_feat

        return data


class BatchFeature(BaseTransform):
    """Assign statistical batch features for each cell."""

    def __init__(self, *, channel: Optional[str] = None, mod: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.mod = mod

    def __call__(self, data):
        # TODO: use get_feature; move mod1 to mod; replace for loop with np
        cells = []
        columns = [
            "cell_mean",
            "cell_std",
            "nonzero_25%",
            "nonzero_50%",
            "nonzero_75%",
            "nonzero_max",
            "nonzero_count",
            "nonzero_mean",
            "nonzero_std",
            "batch",
        ]  # yapf: disable

        ad_input = data.data["mod1"]
        bcl = list(ad_input.obs["batch"])
        print(set(bcl))
        for i, cell in enumerate(ad_input.X):
            cell = cell.toarray()
            nz = cell[np.nonzero(cell)]
            if len(nz) == 0:
                raise ValueError("Error: one cell contains all zero features.")
            cells.append([
                cell.mean(),
                cell.std(),
                np.percentile(nz, 25),
                np.percentile(nz, 50),
                np.percentile(nz, 75),
                cell.max(),
                len(nz) / 1000,
                nz.mean(),
                nz.std(), bcl[i]
            ])

        cell_features = pd.DataFrame(cells, columns=columns)
        batch_source = cell_features.groupby("batch").mean().reset_index()
        batch_list = batch_source.batch.tolist()
        batch_source = batch_source.drop("batch", axis=1).to_numpy().tolist()
        b2i = dict(zip(batch_list, range(len(batch_list))))
        batch_features = []

        for b in ad_input.obs["batch"]:
            batch_features.append(batch_source[b2i[b]])

        batch_features = np.array(batch_features).astype(float)
        data.data["mod1"].obsm["batch_features"] = batch_features
        return data
