import numpy as np
import scipy.sparse as sp

from dance.transforms.base import BaseTransform
from dance.typing import Dict, List, Literal, NormMode, Optional, Union
from dance.utils.matrix import normalize


class ScaleFeature(BaseTransform):
    """Scale the feature matrix in the AnnData object.

    This is an extension of :meth:`scanpy.pp.scale`, allowing split- or batch-wide scaling.

    Parameters
    ----------
    axis
        Axis along which the scaling is performed.
    split_names
        Indicate which splits to perform the scaling independently. If set to 'ALL', then go through all splits
        available in the data.
    batch_key
        Indicate which column in ``.obs`` to use as the batch index to guide the batch-wide scaling.
    mode
        Scaling mode, see :meth:`dance.utils.matrix.normalize` for more information.
    eps
        Correction fact, see :meth:`dance.utils.matrix.normalize` for more information.

    Note
    ----
    The order of checking split- or batch-wide scaling mode is: batch_key > split_names > None (i.e., all).

    """

    _DISPLAY_ATTRS = ("axis", "mode", "eps", "split_names", "batch_key")

    def __init__(
        self,
        *,
        axis: int = 0,
        split_names: Optional[Union[Literal["ALL"], List[str]]] = None,
        batch_key: Optional[str] = None,
        mode: NormMode = "normalize",
        eps: float = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.split_names = split_names
        self.batch_key = batch_key
        self.mode = mode
        self.eps = eps

    def _get_idx_dict(self, data) -> List[Dict[str, List[int]]]:
        batch_key = self.batch_key
        split_names = self.split_names

        if batch_key is not None:
            if split_names is not None:
                raise ValueError("Exactly one of split_names and batch_key can be specified, got: "
                                 f"split_names={split_names!r}, batch_key={batch_key!r}")
            elif batch_key not in (avail_opts := data.data.obs.columns.tolist()):
                raise KeyError(f"{batch_key=!r} not found in `.obs`. Available columns are: {avail_opts}")
            batch_col = data.data.obs[batch_key]
            idx_dict = {f"batch:{i}": np.where(batch_col[0] == i)[0].tolist() for i in batch_col[0].unique()}
            return idx_dict

        if split_names is None:
            idx_dict = {"full": list(range(data.shape[0]))}
        elif isinstance(split_names, str) and split_names == "ALL":
            idx_dict = {f"split:{i}": j for i, j in data._split_idx_dict.items()}
        elif isinstance(split_names, list):
            idx_dict = {f"split:{i}": data.get_split_idx(i) for i in split_names}
        else:
            raise TypeError(f"Unsupported type {type(split_names)} for split_names: {split_names!r}")

        return idx_dict

    def __call__(self, data):
        if isinstance(data.data.X, sp.spmatrix):
            self.logger.warning("Native support for sparse matrix is not implemented yet, "
                                "converting to dense array explicitly.")
            data.data.X = data.data.X.A

        idx_dict = self._get_idx_dict(data)
        for name, idx in idx_dict.items():
            self.logger.info(f"Scaling {name} (n={len(idx):,})")
            data.data.X[idx] = normalize(data.data.X[idx], mode=self.mode, axis=self.axis, eps=self.eps)
