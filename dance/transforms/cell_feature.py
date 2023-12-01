import math

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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


class CellSVD(BaseTransform):

    _DISPLAY_ATTRS = ("n_components", )

    def __init__(self, n_components: int = 400, *, channel: Optional[str] = None, mod: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.n_components = n_components
        self.channel = channel
        self.mod = mod

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy", channel=self.channel, mod=self.mod)
        svd = TruncatedSVD(n_components=self.n_components)

        self.logger.info(f"Start generating cell SVD features {feat.shape} (k={self.n_components})")
        cell_feat = svd.fit_transform(feat)
        evr = svd.explained_variance_ratio_
        self.logger.info(f"Top 10 explained variances: {evr[:10]}")
        self.logger.info(f"Total explained variance: {evr.sum():.2%}")

        data.data.obsm[self.out] = cell_feat

        return data


class CellReduction(BaseTransform):
"""
Provide Three methods of dimensionality reduction. https://github.com/xy-chen16/EnClaSC
 The following is the method：
1EnClaSC
2Seurat v3.0
3scmap
"""
    _DISPLAY_ATTRS = ("n_components", )

    def __init__(self, method: str, n_components: int = 400, *, channel: Optional[str] = None,
                 mod: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.n_components = n_components
        self.channel = channel
        self.mod = mod
        self.method = method

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy", channel=self.channel, mod=self.mod).T
        if self.method == "Seurat":
            data_original = np.exp(feat) - 1
            data_train_mean = np.mean(data_original, 1)
            data_train_var = np.var(data_original, 1)
            data_train_mean_log = np.log(data_train_mean + 1)
            data_train_var_log = np.log(data_train_var + 1)
            ploy_reg = PolynomialFeatures(degree=2)
            x_ = ploy_reg.fit_transform(data_train_mean_log.reshape(-1, 1))
            regr2 = linear_model.LinearRegression()
            regr2.fit(x_, data_train_var_log)
            y_pred = regr2.predict(x_)
            filter_scores = (data_train_var_log - y_pred).reshape(-1)
            ind_romanov = np.argpartition(filter_scores, -self.n_components)[-self.n_components:].tolist()
            filtered_data_train = feat[ind_romanov, :].T
        elif self.method == "EnClaSc":
            feature_number = len(feat)
            zeisel_scores = np.zeros(feature_number) - 100
            data_original = np.exp(feat) - 1
            drop_feature = []
            X1 = []
            Y = []
            X2 = []
            feature_index_dict = []
            for feature in data_original:
                drop_feature.append(np.sum(feature == 0) / feature.shape[0])
            data_train_mean = np.mean(data_original, 1)
            for i in range(len(drop_feature)):
                if (drop_feature[i] != 1 and drop_feature[i] != 0):
                    feature_index_dict.append(i)
                    Y.append(np.log(data_train_mean[i] + 1))
                    X1.append(np.mean(feat[i, :]))
                    X2.append(np.log(drop_feature[i]))
            Y = np.array(Y).reshape(-1, 1)
            X1 = np.array(X1).reshape(-1, 1)
            X2 = np.array(X2).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X2, Y)
            predictions = lr.predict(X2)
            residuals = (Y - predictions).reshape(-1) + (Y - X1).reshape(-1)
            zeisel_scores[feature_index_dict] = residuals
            ind_romanov = np.argpartition(zeisel_scores, -self.n_components)[-self.n_components:].tolist()
            filtered_data_train = feat[ind_romanov, :].T
        elif self.method == "scmap":
            feature_number = len(feat)
            romanov_scores = np.zeros(feature_number) - 100
            data_original = np.exp(feat) - 1
            data_original_mean = np.mean(data_original, 1)
            drop_feature = []
            for feature in data_original:
                drop_feature.append(np.sum(feature == 0) / feature.shape[0])
            feature_index_dict = []
            train_E = []
            train_D = []
            for i in range(len(drop_feature)):
                if (drop_feature[i] != 1 and drop_feature[i] != 0):
                    feature_index_dict.append(i)
                    train_E.append(data_original_mean[i])
                    train_D.append(drop_feature[i])
            train_E = np.array(train_E)
            train_D = np.array(train_D)
            train_E = np.log(train_E + 1).reshape(-1, 1) * math.log(2.7) / math.log(2)
            train_D = np.log(100 * train_D).reshape(-1, 1) * math.log(2.7) / math.log(2)

            lr = LinearRegression()
            lr.fit(train_E, train_D)
            predictions = lr.predict(train_E)
            residuals = (train_D - predictions).reshape(-1)
            #residuals = np.abs(residuals)
            romanov_scores[feature_index_dict] = residuals
            ind_romanov = np.argpartition(romanov_scores, -self.n_components)[-self.n_components:].tolist()
            filtered_data_train = feat[ind_romanov, :].T
        data.data.obsm[self.out] = filtered_data_train
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
