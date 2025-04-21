import logging
from typing import Literal, Optional

import numpy as np
from scipy.sparse import spmatrix
from scipy.stats import expon

from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.typing import Literal, Optional

# @register_preprocessor("split", "entry")
# class CellwiseMaskData(BaseTransform):
#     """Randomly mask data in a cell-wise approach.

#     For every cell that has more than 5 positive counts, mask positive counts according to masking rate and probabiliy
#     generated from distribution.

#     Parameters
#     ----------
#     distr
#         Distribution to generate masks.
#     mask_rate
#         Masking rate.
#     seed:
#         Random seed.
#     Min_gene_counts
#         Minimum number of genes expressed within a below which we do not mask that cell.

#     """

#     _DISPLAY_ATTRS = ("distr", "mask_rate", "seed")

#     def __init__(self, distr: Optional[Literal["exp", "uniform"]] = "exp", mask_rate: Optional[float] = 0.1,
#                  seed: Optional[int] = None, min_gene_counts: int = 5, **kwargs):
#         super().__init__(**kwargs)
#         self.distr = distr
#         self.mask_rate = mask_rate
#         self.seed = seed
#         self.min_gene_counts = min_gene_counts

#     def _get_probs(self, vec):
#         if self.distr == "exp":
#             prob = expon.pdf(vec, 0, 20)
#         elif self.distr == "uniform":
#             prob = np.ones(len(vec))
#         else:
#             raise ValueError(f"Unknown distribution function option {self.distr!r}, "
#                              "available options are: 'exp', 'uniform'")
#         return prob / prob.sum()

#     def __call__(self, data):
#         rng = np.random.default_rng(self.seed)
#         feat = data.get_feature(return_type="sparse")
#         train_mask = np.ones(feat.shape, dtype=bool)

#         for c in range(feat.shape[0]):
#             # Retrieve indices of positive values
#             ind_pos = np.nonzero(feat[c])[-1]
#             cells_c_pos = feat[c, ind_pos]

#             # Get masking probability of each value
#             if cells_c_pos.size > self.min_gene_counts:
#                 prob = self._get_probs(cells_c_pos.toarray()[0])
#                 n_masked = int(np.floor(cells_c_pos.size * self.mask_rate))
#                 if n_masked >= cells_c_pos.size:
#                     self.logger.warning(f"Too many genes masked for cell {c} ({n_masked}/{cells_c_pos.size})")
#                     n_masked = 1 + int(np.floor(0.5 * cells_c_pos.size))

#                 masked_idx = rng.choice(cells_c_pos.size, n_masked, p=prob, replace=False)
#                 train_mask[c, ind_pos[masked_idx]] = False

#         data.data.layers["train_mask"] = train_mask
#         data.data.layers["valid_mask"] = ~train_mask

#         return data


# Assuming BaseTransform and register_preprocessor are defined elsewhere
# For demonstration, let's define dummy versions:
class BaseTransform:

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Process remaining kwargs if needed by the base class
        pass


def register_preprocessor(name, stage):

    def decorator(cls):
        # In a real scenario, this would register the class
        # For now, it just returns the class unmodified
        return cls

    return decorator


# --- Modified Class ---


@register_preprocessor("split", "entry")
class CellwiseMaskData(BaseTransform):
    """Randomly mask data in a cell-wise approach.

    For every cell that has more than `min_gene_counts` positive counts,
    mask positive counts according to `mask_rate` and probability generated
    from the specified distribution.

    The masked entries are assigned to validation and optionally test masks.

    Parameters
    ----------
    distr
        Distribution to generate probabilities for masking counts.
        Higher counts might have different probabilities depending on the distribution.
    mask_rate
        Overall masking rate (proportion of positive counts to mask per cell).
    seed
        Random seed for reproducibility.
    min_gene_counts
        Minimum number of positive counts within a cell below which we do not mask that cell.
    add_test_mask
        If True, the masked entries (determined by `mask_rate`) are further split
        into validation and test sets. Approximately 10% of the masked entries
        go to `valid_mask`, and the remaining 90% go to `test_mask`.
        If False, all masked entries go to `valid_mask`, and `test_mask` will be empty (all False).
    **kwargs
        Additional keyword arguments passed to the base class.

    """

    _DISPLAY_ATTRS = ("distr", "mask_rate", "seed", "min_gene_counts", "add_test_mask")

    def __init__(
            self,
            distr: Optional[Literal["exp", "uniform"]] = "exp",
            mask_rate: Optional[float] = 0.1,
            seed: Optional[int] = None,
            min_gene_counts: int = 5,
            add_test_mask: bool = False,  # New parameter
            **kwargs):
        super().__init__(**kwargs)
        self.distr = distr
        if not 0.0 <= mask_rate <= 1.0:
            raise ValueError(f"mask_rate must be between 0 and 1, got {mask_rate}")
        self.mask_rate = mask_rate
        self.seed = seed
        self.min_gene_counts = min_gene_counts
        self.add_test_mask = add_test_mask  # Store the new parameter

    def _get_probs(self, vec):
        """Calculates sampling probabilities based on the distribution."""
        if self.distr == "exp":
            # Exponential PDF - higher values might be less likely to be masked depending on scale
            # Using scale=20 as in the original code. Adjust if needed.
            prob = expon.pdf(vec, 0, 20)
        elif self.distr == "uniform":
            # Uniform PDF - all positive counts have equal probability of being masked
            prob = np.ones(len(vec))
        else:
            raise ValueError(f"Unknown distribution function option {self.distr!r}, "
                             "available options are: 'exp', 'uniform'")

        # Normalize probabilities if they sum to a positive value
        prob_sum = prob.sum()
        if prob_sum > 1e-9:  # Avoid division by zero if all probs are effectively zero
            return prob / prob_sum
        else:
            # If sum is zero (e.g., vec was empty or pdf returned all zeros), return uniform probability
            # This case should ideally be handled before calling _get_probs, but added as a safeguard.
            self.logger.warning("Probability sum is zero, falling back to uniform probability.")
            return np.ones(len(vec)) / len(vec) if len(vec) > 0 else np.array([])

    def __call__(self, data):
        """Applies the cell-wise masking.

        Parameters
        ----------
        data
            An object containing the feature data (e.g., an AnnData object or similar).
            Requires a method `get_feature(return_type="sparse")` that returns a
            scipy sparse matrix (cells x genes), and allows adding layers via
            `data.data.layers["layer_name"] = mask_array`.

        Returns
        -------
        data
            The input data object with added layers: "train_mask", "valid_mask",
            and "test_mask".

        """
        rng = np.random.default_rng(self.seed)
        # Assuming get_feature returns a CSR or CSC matrix for efficient row slicing
        feat = data.get_feature(return_type="sparse")

        if not isinstance(feat, spmatrix):
            raise TypeError(f"Expected feature data to be a scipy sparse matrix, got {type(feat)}")

        n_cells, n_genes = feat.shape
        train_mask = np.ones((n_cells, n_genes), dtype=bool)
        valid_mask = np.zeros((n_cells, n_genes), dtype=bool)
        test_mask = np.zeros((n_cells, n_genes), dtype=bool)  # Initialize test mask

        for c in range(n_cells):
            # Efficiently get data and indices for the current cell (row)
            cell_slice = feat[c, :]
            ind_pos = cell_slice.indices  # Indices of non-zero elements in this row
            cells_c_pos_values = cell_slice.data  # Values of non-zero elements

            num_positive = len(ind_pos)

            # Only mask if the cell has enough expressed genes
            if num_positive > self.min_gene_counts:
                # Calculate number of entries to mask based on the rate
                n_masked = int(np.floor(num_positive * self.mask_rate))

                # Ensure we don't try to mask more than available or zero items
                if n_masked <= 0:
                    continue  # No masking needed for this cell

                if n_masked >= num_positive:
                    self.logger.warning(f"Mask rate {self.mask_rate} resulted in attempting to mask all "
                                        f"{num_positive} positive counts for cell {c}. Reducing mask count.")
                    # Mask roughly half instead of all/too many
                    n_masked = 1 + int(np.floor(0.5 * num_positive))

                # Get masking probability for each positive value if needed
                if self.distr == "exp":
                    # Need the actual values to calculate exp probability
                    prob = self._get_probs(cells_c_pos_values)
                else:  # Uniform distribution
                    prob = None  # np.random.choice uses uniform sampling if p is None

                # Check if probabilities are valid before using them
                if prob is not None and (len(prob) != num_positive or not np.isclose(prob.sum(), 1.0)):
                    self.logger.warning(f"Invalid probabilities calculated for cell {c}. Falling back to uniform.")
                    prob = None  # Fallback to uniform if probabilities are problematic

                # Choose indices to mask *from the set of positive indices*
                try:
                    masked_relative_idx = rng.choice(
                        num_positive,  # Choose from 0 to num_positive-1
                        size=n_masked,
                        p=prob,
                        replace=False  # Ensure unique indices are masked
                    )
                except ValueError as e:
                    # This might happen if probabilities don't sum to 1, or num_positive mismatch
                    self.logger.error(f"Error during rng.choice for cell {c}: {e}. Skipping masking for this cell.")
                    continue

                # Get the absolute column indices in the feature matrix
                masked_absolute_indices = ind_pos[masked_relative_idx]

                # Mark these chosen indices as False in the training mask
                train_mask[c, masked_absolute_indices] = False

                # Now, decide where these masked entries go (validation or test)
                if self.add_test_mask:
                    # Split the masked_absolute_indices into validation and test sets (10% validation, 90% test)
                    n_total_masked_in_cell = len(masked_absolute_indices)

                    if n_total_masked_in_cell > 0:
                        # Calculate the number of validation samples (approx 10%)
                        # Use rounding, ensure at least 1 if possible (unless total is 0)
                        n_valid = int(np.round(n_total_masked_in_cell * 0.1))
                        n_valid = max(
                            1, n_valid
                        ) if n_total_masked_in_cell > 1 else n_total_masked_in_cell  # Ensure at least 1 unless only 1 total

                        # Shuffle the indices before splitting to ensure randomness
                        shuffled_masked_indices = rng.permutation(masked_absolute_indices)

                        # Assign to validation and test masks
                        valid_indices_c = shuffled_masked_indices[:n_valid]
                        test_indices_c = shuffled_masked_indices[n_valid:]

                        valid_mask[c, valid_indices_c] = True
                        test_mask[c, test_indices_c] = True

                    # If n_total_masked_in_cell is 0 (shouldn't happen if n_masked > 0), do nothing

                else:
                    # Original behavior: all masked entries go to the validation mask
                    valid_mask[c, masked_absolute_indices] = True

        # Store the masks in the data object's layers
        # Ensure the data structure supports adding layers like this
        if not hasattr(data, 'data') or not hasattr(data.data, 'layers'):
            raise AttributeError("Input data object does not have the expected structure 'data.layers'")

        data.data.layers["train_mask"] = train_mask
        data.data.layers["valid_mask"] = valid_mask
        data.data.layers["test_mask"] = test_mask  # Always add test_mask, even if it's all False

        # Log mask statistics
        n_total = feat.shape[0] * feat.shape[1]
        n_train = train_mask.sum()
        n_valid = valid_mask.sum()
        n_test = test_mask.sum()
        n_masked_total = n_valid + n_test
        self.logger.info(f"Masking complete. Total elements: {n_total}")
        self.logger.info(f"  Train mask: {n_train} elements ({n_train/n_total:.2%})")
        if self.add_test_mask:
            self.logger.info(f"  Valid mask: {n_valid} elements ({n_valid/n_total:.4%})")
            self.logger.info(f"  Test mask:  {n_test} elements ({n_test/n_total:.4%})")
            if n_masked_total > 0:
                self.logger.info(f"  Validation split of masked: {n_valid / n_masked_total:.2%}")
                self.logger.info(f"  Test split of masked:       {n_test / n_masked_total:.2%}")
        else:
            self.logger.info(f"  Valid mask: {n_valid} elements ({n_valid/n_total:.4%}) (Test mask not created)")

        return data


@register_preprocessor("split", "entry")
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
