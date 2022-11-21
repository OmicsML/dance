from copy import deepcopy

import numpy as np
import torch
from anndata import AnnData

from dance import logger
from dance.typing import CellIdxType, Dict, FeatType, List, Optional, ReturnedFeat, Sequence, Tuple


class Data:
    """Base data object."""

    def __init__(self, x: Optional[AnnData] = None, y: Optional[AnnData] = None, train_size: Optional[int] = None,
                 val_size: int = 0, test_size: int = -1, ensure_cell_aligned: bool = True):
        """Initialize data object.

        Parameters
        ----------
        x : AnnData
            Cell features.
        y : AnnData
            Cell labels.
        train_size
            Number of cells to be used for training. If not specified, not splits will be generated.
        val_size
            Number of cells to be used for validation. If set to -1, use what's left from training and testing.
        test_size
            Number of cells to be used for testing. If set to -1, used what's left from training and validation.
        ensure_cell_aligned
            If set to True, then check for the consistency between the indeices of x and y.

        """
        self._split_idx_dict: Dict[str, Sequence[CellIdxType]] = {}

        self._x = x or AnnData()
        self._y = y

        if ensure_cell_aligned and y:
            if (x_size := x.shape[0]) != (y_size := y.shape[0]):
                raise ValueError(f"Mismatched number of samples between x (n={x_size:,}) and y (n={y_size:,}).")
            elif (num_diff := (x.obs.index != y.obs.index).sum()) > 0:
                raise IndexError(f"{num_diff:,} out of {x_size:,} entries have mismached indices between x and y. ",
                                 "Set `ensure_cell_aligned` to False if you do not wish to align x and y.")

        self._setup_splits(train_size, val_size, test_size)

    def _setup_splits(self, train_size: Optional[int], val_size: int, test_size: int):
        if train_size is None:
            return
        elif any(not isinstance(i, int) for i in (train_size, val_size, test_size)):
            raise TypeError("Split sizes must be of type int")

        split_names = ["train", "val", "test"]
        split_sizes = np.array((train_size, val_size, test_size))

        # Only one -1 (complementary size) is allowed
        if (split_sizes == -1).sum() > 1:
            raise ValueError("Only one split can be specified as -1")

        # Each size must be bounded between -1 and the data size
        data_size = self.x.shape[0]
        for name, size in zip(split_names, split_sizes):
            if size < -1:
                raise ValueError(f"{name} must be integer no less than -1, got {size!r}")
            elif size > data_size:
                raise ValueError(f"{name}={size:,} exceeds total number of samples {data_size:,}")

        # Sum of sizes must be bounded by the data size
        if (tot_size := split_sizes.clip(0).sum()) > data_size:
            raise ValueError(f"Total size {tot_size:,} exceeds total number of sampels {data_size:,}")

        logger.debug(f"Split sizes before conversion: {split_sizes.tolist()}")
        split_sizes[split_sizes == -1] = data_size - split_sizes.clip(0).sum()
        logger.debug(f"Split sizes after conversion: {split_sizes.tolist()}")

        all_idx = self.x.obs.index.values
        split_thresholds = split_sizes.cumsum()
        for i, split_name in enumerate(split_names):
            start = split_thresholds[i - 1] if i > 0 else 0
            end = split_thresholds[i]
            if end - start > 0:  # skip empty split
                self._split_idx_dict[split_name] = all_idx[start:end].tolist()

    def __getitem__(self, idx) -> Tuple[AnnData, AnnData]:
        sliced_x = self.x[idx]
        sliced_y = self.y[idx]
        return sliced_x, sliced_y

    @property
    def x(self) -> AnnData:
        return self._x

    @property
    def y(self) -> Optional[AnnData]:
        return self._y

    @property
    def num_cells(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def cells(self) -> List[str]:
        return self.x.obs.index.tolist()

    @property
    def train_idx(self) -> Sequence[CellIdxType]:
        return self.get_split_idx("train", error_on_miss=False)

    @property
    def val_idx(self) -> Sequence[CellIdxType]:
        return self.get_split_idx("val", error_on_miss=False)

    @property
    def test_idx(self) -> Sequence[CellIdxType]:
        return self.get_split_idx("test", error_on_miss=False)

    def copy(self):
        return deepcopy(self)

    def set_split_idx(self, split_name: str, split_idx: Sequence[CellIdxType]):
        """Set cell indices for a particular split.

        Parameters
        ----------
        split_name
            Name of the split to set.
        split_idx
            Indices of the cells to be used in this split.

        """
        self._split_idx_dict[split_name] = split_idx

    def get_split_idx(self, split_name: str, error_on_miss: bool = False):
        """Obtain cell indices for a particular split.

        Parameters
        ----------
        split_name : str
            Name of the split to retrieve.
        error_on_miss
            If set to True, raise KeyError if the queried split does not exit, otherwise return None.

        """
        if split_name in self._split_idx_dict:
            return self._split_idx_dict[split_name]
        elif error_on_miss:
            raise KeyError(f"Unknown split {split_name!r}. Please set the split inddices via set_split_idx first.")
        else:
            return None

    def _get_feat(self, feat_name: str, split_name: Optional[str], return_type: FeatType = "numpy",
                  layer: Optional[str] = None, channel: Optional[str] = None):
        if (layer is not None) and (channel is not None):
            raise ValueError(f"Cannot specify layer ({layer!r}) and channel ({channel!r}) simmultaneously.")

        if split_name is None:
            feat = getattr(self, feat_name)
        elif split_name in self._split_idx_dict:
            idx = self.get_split_idx(split_name)
            feat = getattr(self, feat_name)[idx]
        else:
            raise KeyError(f"Unknown split {split_name!r}, available options are {list(self._split_idx_dict)}")

        # Directly return anndata view
        if return_type == "anndata":
            return feat

        # Extract features from anndata
        if layer is not None:
            feat = feat.layers[layer].X
        elif channel is not None:
            feat = feat.obsm[channel]
        else:
            feat = feat.X

        try:  # convert sparse array to dense array
            feat = feat.toarray()
        except AttributeError:
            pass

        if return_type == "torch":
            feat = torch.from_numpy(feat)
        elif return_type != "numpy":
            raise ValueError(f"Unknown return_type {return_type!r}")

        return feat

    def get_x(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", layer: Optional[str] = None,
              channel: Optional[str] = None) -> ReturnedFeat:
        """Retrieve cell features from a particular split."""
        return self._get_feat("x", split_name, return_type, layer, channel)

    def get_y(self, split_name: Optional[str] = None, return_type: FeatType = "numpy") -> ReturnedFeat:
        """Retrieve cell labels from a particular split."""
        return self._get_feat("y", split_name, return_type)

    def get_x_y(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", layer: Optional[str] = None,
                channel: Optional[str] = None) -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from a particular split.

        Parameters
        ----------
        split_name
            Name of the split to retrieve. If not set, return all.
        return_type
            How should the features be returned. **numpy**: return as a numpy array; **torch**: return as a torch
            tensor; **anndata**: return as an anndata object.
        layer
            Return a particular layer as features.
        channel
            Return a particular obsm channel as features.

        Notes
        -----
        If both `layer` and `channel` are not specified (default), then return the default layer as features.

        """
        x = self.get_x(split_name, return_type, layer, channel)
        y = self.get_y(split_name, return_type)
        return x, y

    def get_train_data(self, return_type: FeatType = "numpy", layer: Optional[str] = None,
                       channel: Optional[str] = None) -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from the 'train' split."""
        return self.get_x_y("train", return_type, layer, channel)

    def get_val_data(self, return_type: FeatType = "numpy", layer: Optional[str] = None,
                     channel: Optional[str] = None) -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from the 'val' split."""
        return self.get_x_y("val", return_type, layer, channel)

    def get_test_data(self, return_type: FeatType = "numpy", layer: Optional[str] = None,
                      channel: Optional[str] = None) -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from the 'test' split."""
        return self.get_x_y("test", return_type, layer, channel)
