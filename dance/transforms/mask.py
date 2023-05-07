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
    mask_type
        Imputation mask type. Supports ['mcar', 'mar'].
    valid_mask_rate
        Validation masking rate.
    test_mask_rate
        Testing masking rate.
    seed:
        Random seed.
    Min_gene_counts
        Minimum number of genes expressed within a below which we do not mask that cell.

    """

    _DISPLAY_ATTRS = ("mask_type", "valid_mask_rate", "test_mask_rate", "seed", "min_gene_counts")

    def __init__(self, mask_type: Optional[str] = "mar", valid_mask_rate: Optional[float] = 0.1, 
                 test_mask_rate: Optional[float] = 0.1, seed: Optional[int] = None, 
                 min_gene_counts: int = 5, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= valid_mask_rate < 1, "valid_mask_rate should be in [0, 1)"
        assert 0 < test_mask_rate < 1, "test_mask_rate should be in (0, 1)"
        assert 0 < valid_mask_rate + test_mask_rate < 1, "Total masking rate should be in (0, 1)"
        self.valid_mask_rate = valid_mask_rate
        self.test_mask_rate = test_mask_rate
        self.seed = seed
        self.min_gene_counts = min_gene_counts
        
        self.mask_type = mask_type
        if mask_type == "mcar":
            self.distr = "uniform"
        elif mask_type == "mar":
            self.distr = "exp"
        else:
            raise NotImplementedError(f"Expect mask_type in ['mar', 'mcar'], but found {mask_type}")

    def _get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)

    def __call__(self, data):
        rng = np.random.default_rng(self.seed)
        feat = data.get_feature(return_type="default")
        train_mask = np.ones(feat.shape, dtype=bool)
        valid_mask = np.zeros(feat.shape, dtype=bool)
        test_mask = np.zeros(feat.shape, dtype=bool)

        for c in range(feat.shape[0]):
            # Retrieve indices of positive values
            ind_pos = np.nonzero(feat[c])[-1]
            cells_c_pos = feat[c, ind_pos]

            # Get masking probability of each value
            if cells_c_pos.size > self.min_gene_counts:
                mask_prob = self._get_probs(cells_c_pos.toarray()[0])
                mask_prob = mask_prob / sum(mask_prob)
                n_test = int(np.floor(cells_c_pos.size * self.test_mask_rate))
                n_valid = int(np.floor(cells_c_pos.size * self.valid_mask_rate))
                if n_test + n_valid >= cells_c_pos.size:
                    self.logger.warning(f"Too many genes masked for cell {c} ({n_test + n_valid}/{cells_c_pos.size})")
                    n_test -= 1
                    n_valid -= 1
                
                idx_mask = np.ones(len(ind_pos), dtype=bool)
                test_idx = rng.choice(np.arange(len(ind_pos)), n_test, p=mask_prob, replace=False)
                train_mask[c, ind_pos[test_idx]] = False
                test_mask[c, ind_pos[test_idx]] = True
                if self.valid_mask_rate > 0:
                    idx_mask[test_idx] = False
                    masked_mask_prob = mask_prob[idx_mask] / sum(mask_prob[idx_mask])
                    valid_idx = rng.choice(np.arange(len(ind_pos))[idx_mask], n_valid, p=masked_mask_prob, replace=False)
                    train_mask[c, ind_pos[valid_idx]] = False
                    valid_mask[c, ind_pos[valid_idx]] = True

        data.data.layers["train_mask"] = train_mask
        data.data.layers["valid_mask"] = valid_mask
        data.data.layers["test_mask"] = test_mask

        return data


class MaskData(BaseTransform):
    """Randomly mask data.

    Randomly mask positive counts according to masking type.

    Parameters
    ----------
    mask_type
        Imputation masking type. Supports ['mcar', 'mar'].
    valid_mask_rate
        Validation masking rate.
    test_mask_rate
        Testing masking rate.
    seed:
        Random seed.

    """

    _DISPLAY_ATTRS = ("mask_type", "valid_mask_rate", "test_mask_rate", "seed")

    def __init__(self, mask_type: Optional[str] = "mar", valid_mask_rate: Optional[float] = 0.1, 
                 test_mask_rate: Optional[float] = 0.1, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= valid_mask_rate < 1, "valid_mask_rate should be in [0, 1)"
        assert 0 < test_mask_rate < 1, "test_mask_rate should be in (0, 1)"
        assert 0 < valid_mask_rate + test_mask_rate < 1, "Total masking rate should be in (0, 1)"
        self.valid_mask_rate = valid_mask_rate
        self.test_mask_rate = test_mask_rate
        self.seed = seed
        if mask_type == "mcar":
            self.distr = "uniform"
        elif mask_type == "mar":
            self.distr = "exp"
        else:
            raise NotImplementedError(f"Expect mask_type in ['mar', 'mcar'], but found {mask_type}")

    def _get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)

    def __call__(self, data):
        rng = np.random.default_rng(self.seed)
        feat = data.get_feature(return_type="default")
        train_mask = np.ones(feat.shape, dtype=bool)
        valid_mask = np.zeros(feat.shape, dtype=bool)
        test_mask = np.zeros(feat.shape, dtype=bool)
        row, col = np.nonzero(feat)
        nonzero_counts = np.array(feat[row, col])[0]
        num_nonzeros = len(row)
        n_test = int(np.floor(num_nonzeros * self.test_mask_rate))
        n_valid = int(np.floor(num_nonzeros * self.valid_mask_rate))
        idx_mask = np.ones(num_nonzeros, dtype=bool)

        # Randomly mask positive counts according to masking probability.
        mask_prob = self._get_probs(nonzero_counts)
        mask_prob = mask_prob / sum(mask_prob)
        test_idx = rng.choice(np.arange(num_nonzeros), n_test, p=mask_prob, replace=False)
        train_mask[row[test_idx], col[test_idx]] = False
        test_mask[row[test_idx], col[test_idx]] = True  
        if self.valid_mask_rate > 0:
            idx_mask[test_idx] = False
            masked_mask_prob = mask_prob[idx_mask] / sum(mask_prob[idx_mask])
            valid_idx = rng.choice(np.arange(num_nonzeros)[idx_mask], n_valid, p=masked_mask_prob, replace=False)
            train_mask[row[valid_idx], col[valid_idx]] = False  
            valid_mask[row[valid_idx], col[valid_idx]] = True        

        data.data.layers["train_mask"] = train_mask
        data.data.layers["valid_mask"] = valid_mask
        data.data.layers["test_mask"] = test_mask

        return data
