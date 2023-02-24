import numpy as np
import scipy.sparse as sp
from scipy.stats import expon

from dance.transforms.base import BaseTransform
from dance.typing import Optional


class CellwiseMaskData(BaseTransform):
    """Randomly mask data in a cell-wise approach.

    For every cell that has more than 5 positive counts, mask positive counts according to masking rate and probabiliy
    generated from distribution.

    Parameters
    ----------
    distr
        Distribution to generate masks.
    mask_rate
        Masking rate.
    seed:
        Random seed.

    """

    _DISPLAY_ATTRS = ("distr", "mask_rate", "seed")

    def __init__(self, distr: Optional[str] = "exp", mask_rate: Optional[float] = 0.1, seed: Optional[int] = 1, **kwargs):
        super().__init__(**kwargs)
        self.distr = distr
        self.mask_rate = mask_rate
        self.seed = seed
    
    def _get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)

    def __call__(self, data):
        rng = np.random.default_rng(self.seed)
        feat = data.get_feature(return_type="default")
        feat_raw = data.get_feature(return_type="default", channel_type="raw_X")
        trainMask = np.ones_like(feat).astype(bool)
        for c in range(feat.shape[0]):
            cells_c = feat[c, :]
            # Retrieve indices of positive values
            ind_pos = np.arange(feat.shape[1])[cells_c > 0]
            cells_c_pos = cells_c[ind_pos]

            # Get masking probability of each value
            if cells_c_pos.size > 5:
                probs = self._get_probs(cells_c_pos)
                n_masked = int(np.floor(len(cells_c_pos) * self.mask_rate))
                if n_masked >= cells_c_pos.size:
                    print("Warning: too many cells masked for gene {} ({}/{})".format(c, n_masked, cells_c_pos.size))
                    n_masked = 1 + int(0.5 * cells_c_pos.size)

                masked_idx = rng.choice(cells_c_pos.size, n_masked, p=probs / probs.sum(), replace=False)
                trainMask[c, ind_pos[sorted(masked_idx)]] = False
        
        data.data.layers["train_mask"] = trainMask
        data.data.layers["valid_mask"] = ~trainMask

        return data


class MaskData(BaseTransform):
    """Randomly mask data.

    Randomly mask positive counts according to masking rate.

    Parameters
    ----------
    mask_rate
        Masking rate.
    seed:
        Random seed.

    """

    _DISPLAY_ATTRS = ("mask_rate", "seed")

    def __init__(self, mask_rate: Optional[float] = 0.1, seed: Optional[int] = 1, **kwargs):
        super().__init__(**kwargs)
        self.mask_rate = mask_rate
        self.seed = seed
    
    def _get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)

    def __call__(self, data):
        rng = np.random.default_rng(self.seed)
        feat = data.get_feature(return_type="default")
        feat_raw = data.get_feature(return_type="default", channel_type="raw_X")
        trainMask = np.ones_like(feat).astype(bool)

        row, col = np.nonzero(feat)
        feat_pos = feat[row, col]
        
        # Randomly mask positive counts according to masking rate.
        num_valid = int(np.floor(len(row) * self.mask_rate))
        num_train = len(row) - num_valid
        all_features_idx = np.arange(len(feat_pos))
        rng.shuffle(all_features_idx)
        train_data_idx = all_features_idx[:num_train]
        valid_data_idx = all_features_idx[num_train:(num_train+num_valid)]
        
        for i in train_data_idx:
            trainMask[row[i], col[i]] = False
        
        data.data.layers["train_mask"] = trainMask
        data.data.layers["valid_mask"] = ~trainMask

        return data