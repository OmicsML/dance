import numpy as np
from scipy.stats import expon

from dance.transforms.base import BaseTransform
from dance.typing import Literal, Optional


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
    Min_gene_counts
        Minimum number of genes expressed within a below which we do not mask that cell.

    """

    _DISPLAY_ATTRS = ("distr", "mask_rate", "seed")

    def __init__(self, distr: Optional[Literal["exp", "uniform"]] = "exp", mask_rate: Optional[float] = 0.1,
                 seed: Optional[int] = None, min_gene_counts: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.distr = distr
        self.mask_rate = mask_rate
        self.seed = seed
        self.min_gene_counts = min_gene_counts

    def _get_probs(self, vec):
        if self.distr == "exp":
            prob = expon.pdf(vec, 0, 20)
        elif self.distr == "uniform":
            prob = np.ones(len(vec))
        else:
            raise ValueError(f"Unknown distribution function option {self.distr!r}, "
                             "available options are: 'exp', 'uniform'")
        return prob / prob.sum()

    def __call__(self, data):
        rng = np.random.default_rng(self.seed)
        feat = data.get_feature(return_type="default")
        train_mask = np.ones(feat.shape, dtype=bool)

        for c in range(feat.shape[0]):
            # Retrieve indices of positive values
            ind_pos = np.nonzero(feat[c])[-1]
            cells_c_pos = feat[c, ind_pos]

            # Get masking probability of each value
            if cells_c_pos.size > self.min_gene_counts:
                prob = self._get_probs(cells_c_pos.toarray()[0])
                n_masked = int(np.floor(cells_c_pos.size * self.mask_rate))
                if n_masked >= cells_c_pos.size:
                    self.logger.warning(f"Too many genes masked for cell {c} ({n_masked}/{cells_c_pos.size})")
                    n_masked = 1 + int(np.floor(0.5 * cells_c_pos.size))

                masked_idx = rng.choice(cells_c_pos.size, n_masked, p=prob, replace=False)
                train_mask[c, ind_pos[masked_idx]] = False

        data.data.layers["train_mask"] = train_mask
        data.data.layers["valid_mask"] = ~train_mask

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

    def __init__(self, mask_rate: Optional[float] = 0.1, seed: Optional[int] = None, **kwargs):
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
        train_mask = np.ones(feat.shape, dtype=bool)
        row, col = np.nonzero(feat)
        num_nonzero = len(row)

        # Randomly mask positive counts according to masking rate.
        num_train = num_nonzero - int(np.floor(num_nonzero * self.mask_rate))
        mask_idx = rng.choice(num_nonzero, size=num_train, replace=False)
        train_mask[row[mask_idx], col[mask_idx]] = False

        data.data.layers["train_mask"] = train_mask
        data.data.layers["valid_mask"] = ~train_mask

        return data
