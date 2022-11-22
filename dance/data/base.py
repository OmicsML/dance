from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch

from dance import logger
from dance.typing import Any, Dict, FeatType, List, Optional, Sequence, Tuple


class BaseData(ABC):
    """Base data object."""

    def __init__(self, data: Any, train_size: Optional[int] = None, val_size: int = 0, test_size: int = -1):
        """Initialize data object.

        Parameters
        ----------
        data
            Cell data.
        train_size
            Number of cells to be used for training. If not specified, not splits will be generated.
        val_size
            Number of cells to be used for validation. If set to -1, use what's left from training and testing.
        test_size
            Number of cells to be used for testing. If set to -1, used what's left from training and validation.

        """
        super().__init__()

        self._data = data

        # TODO: move _split_idx_dict into data.uns
        self._split_idx_dict: Dict[str, Sequence[int]] = {}
        self._setup_splits(train_size, val_size, test_size)

        if "dance_config" not in self._data.uns:
            self._data.uns["dance_config"] = dict()

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
        data_size = self.num_cells
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

        split_thresholds = split_sizes.cumsum()
        for i, split_name in enumerate(split_names):
            start = split_thresholds[i - 1] if i > 0 else 0
            end = split_thresholds[i]
            if end - start > 0:  # skip empty split
                self._split_idx_dict[split_name] = list(range(start, end))

    def __getitem__(self, idx: Sequence[int]) -> Any:
        return self.data[idx]

    @property
    def data(self):
        return self._data

    @property
    @abstractmethod
    def x(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self):
        raise NotImplementedError

    @property
    def config(self) -> Dict[str, Any]:
        return self._data.uns["dance_config"]

    def set_config(self, **kwargs):
        # TODO: need to have some control about validity of configs
        self.config.update(kwargs)

    @property
    def num_cells(self) -> int:
        return self.data.shape[0]

    @property
    def num_features(self) -> int:
        return self.data.shape[1]

    @property
    def cells(self) -> List[str]:
        return self.data.obs.index.tolist()

    @property
    def train_idx(self) -> Sequence[int]:
        return self.get_split_idx("train", error_on_miss=False)

    @property
    def val_idx(self) -> Sequence[int]:
        return self.get_split_idx("val", error_on_miss=False)

    @property
    def test_idx(self) -> Sequence[int]:
        return self.get_split_idx("test", error_on_miss=False)

    def copy(self):
        return deepcopy(self)

    def set_split_idx(self, split_name: str, split_idx: Sequence[int]):
        """Set cell indices for a particular split.

        Parameters
        ----------
        split_name
            Name of the split to set.
        split_idx
            Indices to be used in this split.

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

    def _get_data(self, name: str, split_name: Optional[str], return_type: FeatType = "numpy", **kwargs):
        out = getattr(self, name)

        if hasattr(out, "toarray"):  # convert sparse array to dense numpy array
            out = out.toarray()
        elif hasattr(out, "to_numpy"):  # convert dataframe to numpy array
            out = out.to_numpy()

        if split_name in self._split_idx_dict:
            idx = self.get_split_idx(split_name)
            out = out[idx]
        elif split_name is not None:
            raise KeyError(f"Unknown split {split_name!r}, available options are {list(self._split_idx_dict)}")

        if return_type == "torch":
            out = torch.from_numpy(out)
        elif return_type != "numpy":
            raise ValueError(f"Unknown return_type {return_type!r}")

        return out

    def get_x(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", **kwargs) -> Any:
        """Retrieve cell features from a particular split."""
        return self._get_data("x", split_name, return_type, **kwargs)

    def get_y(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", **kwargs) -> Any:
        """Retrieve cell labels from a particular split."""
        return self._get_data("y", split_name, return_type, **kwargs)

    def get_x_y(
        self, split_name: Optional[str] = None, return_type: FeatType = "numpy", x_kwargs: Dict[str, Any] = dict(),
        y_kwargs: Dict[str, Any] = dict()
    ) -> Tuple[Any, Any]:
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
        x = self.get_x(split_name, return_type, **x_kwargs)
        y = self.get_y(split_name, return_type, **y_kwargs)
        return x, y

    def get_train_data(
        self, return_type: FeatType = "numpy", x_kwargs: Dict[str, Any] = dict(), y_kwargs: Dict[str, Any] = dict()
    ) -> Tuple[Any, Any]:
        """Retrieve cell features and labels from the 'train' split."""
        return self.get_x_y("train", return_type, x_kwargs, y_kwargs)

    def get_val_data(
        self, return_type: FeatType = "numpy", x_kwargs: Dict[str, Any] = dict(), y_kwargs: Dict[str, Any] = dict()
    ) -> Tuple[Any, Any]:
        """Retrieve cell features and labels from the 'val' split."""
        return self.get_x_y("val", return_type, x_kwargs, y_kwargs)

    def get_test_data(
        self, return_type: FeatType = "numpy", x_kwargs: Dict[str, Any] = dict(), y_kwargs: Dict[str, Any] = dict()
    ) -> Tuple[Any, Any]:
        """Retrieve cell features and labels from the 'test' split."""
        return self.get_x_y("test", return_type, x_kwargs, y_kwargs)

    def concat(self, data, merge_splits: bool = True):
        raise NotImplementedError


class Data(BaseData):

    @property
    def x(self):
        # TODO: check validity when setting up
        channel = self.config.get("feature_channel")
        layer = self.config.get("feature_layer")
        if (channel is not None) and (layer is not None):
            raise ValueError(f"Cannot specify feature layer ({layer!r}) and channel ({channel!r}) simmultaneously.")
        elif channel is not None:
            return self.data.obsm[channel]
        elif layer is not None:
            return self.data.layers[layer].X
        else:
            return self.data.X

    @property
    def y(self):
        if (channel := self.config.get("label_channel")) is None:
            raise ValueError("Label channel has not been specified yet.")
        else:
            return self.data.obsm[channel]
