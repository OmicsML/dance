import numpy as np
from torch.utils.data import DataLoader

from dance.transforms.base import BaseTransform
from dance.typing import Optional


class GeneHoldout(BaseTransform):
    """Progressively hold out genes for DeepImpute

    Split genes into target batches. For every target gene in one batch, refer to the genes that are not in
    this batch and select predictor genes with high covariance with target gene.

    Parameters
    ----------
    n_top
        Number of predictor genes per target gene.
    batch_size
        Target batch size.

    """

    _DISPLAY_ATTRS = ("batch_size", "n_top")

    def __init__(self, n_top: int = 5, batch_size: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.n_top = n_top
        self.batch_size = batch_size

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy")
        batch_loader = DataLoader(range(feat.shape[1]), batch_size=self.batch_size)
        targets = []
        for _, batch in enumerate(batch_loader):
            targets.append(batch.int().numpy())

        # Use covariance to select predictors
        covariance_matrix = np.cov(feat, rowvar=False)
        predictors = []
        for i, targs in enumerate(targets):
            genes_not_in_target = np.setdiff1d(range(feat.shape[1]), targs)
            subMatrix = covariance_matrix[targs][:, genes_not_in_target]
            sorted_idx = np.argsort(-subMatrix, axis=1)
            preds = genes_not_in_target[sorted_idx[:, :self.n_top].flatten()]
            predictors.append(np.unique(preds).astype(int))

        data.data.uns["targets"] = targets
        data.data.uns["predictors"] = predictors

        return data
