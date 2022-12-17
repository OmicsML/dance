import numpy as np
from sklearn.decomposition import PCA

from dance.data.base import BaseData
from dance.transforms.base import BaseTransform
from dance.typing import Optional
from dance.utils.matrix import normalize


class WeightedFeaturePCA(BaseTransform):
    """Compute the weighted gene PCA as cell features.

    Given a gene expression matrix of dimension (cell x gene), the gene PCA is first compured. Then, the representation
    of each cell is computed by taking the weighted sum of the gene PCAs based on that cell's gene expression values.

    """

    def __init__(self, n_components: int = 400, split_name: Optional[str] = None, **kwargs):
        """Initialize WeightedFeaturePCA.

        Parameters
        ----------
        n_components
            Number of PCs to use.
        split_name
            Which split to use to compute the gene PCA. If not set, use all data.

        """
        super().__init__(**kwargs)

        self.n_components = n_components
        self.split_name = split_name

    def __call__(self, data: BaseData) -> BaseData:
        feat = data.get_x(self.split_name)  # cell x genes
        gene_pca = PCA(n_components=self.n_components)

        self.logger.info(f"Start decomposing {self.split_name} features {feat.shape}")
        gene_feat = gene_pca.fit_transform(feat.T)  # decompose into gene features
        self.logger.info(f"Total explained variance: {gene_pca.explained_variance_ratio_.sum():.2%}")

        x = data.get_x()
        cell_feat = normalize(x, mode="normalize", axis=1) @ gene_feat
        data.data.obsm[self.out] = cell_feat.astype(np.float32)
        data.data.varm[self.out] = gene_feat.astype(np.float32)

        return data
