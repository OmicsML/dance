import numpy as np

from dance.transforms.base import BaseTransform
from dance.typing import Optional


class GeneHoldout(BaseTransform):
    """Progressively hold out genes for DeepImpute.

    Split genes into target batches. For every target gene in one batch, refer to the genes that are not in
    this batch and select predictor genes with high covariance with target gene.

    Parameters
    ----------
    n_top
        Number of predictor genes per target gene.
    batch_size
        Target batch size.
    random_state
        Random state.

    """

    _DISPLAY_ATTRS = ("batch_size", "n_top")

    def __init__(self, n_top: int = 5, batch_size: int = 512, random_state: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.n_top = n_top
        self.batch_size = batch_size
        self.random_state = random_state

    def __call__(self, data):
        rng = np.random.default_rng(self.random_state)
        feat = data.get_feature(return_type="numpy")
        targets = np.split(rng.permutation(feat.shape[1]), range(self.batch_size, feat.shape[1], self.batch_size))

        # Use covariance to select predictors
        covariance_matrix = np.cov(feat, rowvar=False)
        predictors = []
        for targs in targets:
            genes_not_in_target = np.setdiff1d(range(feat.shape[1]), targs)
            subMatrix = covariance_matrix[targs][:, genes_not_in_target]
            sorted_idx = np.argsort(-subMatrix, axis=0)
            preds = genes_not_in_target[sorted_idx[:self.n_top].flatten()]
            predictors.append(np.unique(preds))

        data.data.uns["targets"] = targets
        data.data.uns["predictors"] = predictors

        return data
