import itertools
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
from anndata import AnnData
from mudata import MuData

from dance import logger
from dance.typing import Any, Dict, FeatType, Iterator, List, Optional, Sequence, Tuple, Union


def _ensure_iter(val: Optional[Union[List[str], str]]) -> Iterator[Optional[str]]:
    if val is None:
        val = itertools.repeat(None)
    elif isinstance(val, str):
        val = [val]
    elif not isinstance(val, list):
        raise TypeError(f"Input to _ensure_iter must be list, str, or None. Got {type(val)}.")
    return val


def _check_types_and_sizes(types, sizes):
    if len(types) == 0:
        return
    elif len(types) > 1:
        raise TypeError(f"Found mixed types: {types}. Input configs must be either all str or all lists.")
    elif ((type_ := types.pop()) == list) and (len(sizes) > 1):
        raise ValueError(f"Found mixed sizes lists: {sizes}. Input configs must be of same length.")
    elif type_ not in (list, str):
        raise TypeError(f"Unknownn type {type_} found in config.")


class BaseData(ABC):
    """Base data object."""

    _FEATURE_CONFIGS: List[str] = ["feature_mod", "feature_channel", "feature_channel_type"]
    _LABEL_CONFIGS: List[str] = ["label_mod", "label_channel", "label_channel_type"]
    _DATA_CHANNELS: List[str] = ["obsm", "varm", "obsp", "varp", "layers", "uns"]

    def __init__(self, data: Union[AnnData, MuData], train_size: Optional[int] = None, val_size: int = 0,
                 test_size: int = -1):
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

    def _setup_splits(self, train_size: Optional[Union[int, str]], val_size: int, test_size: int):
        if train_size is None:
            return
        elif isinstance(train_size, str) and train_size.lower() == "all":
            train_size = -1
            val_size = test_size = 0
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
        # Check config key validity
        all_configs = set(self._FEATURE_CONFIGS + self._LABEL_CONFIGS)
        if (unknown_options := set(kwargs).difference(all_configs)):
            raise KeyError(f"Unknown config option(s): {unknown_options}, available options are: {all_configs}")

        feature_configs = [j for i, j in kwargs.items() if i in self._FEATURE_CONFIGS]
        label_configs = [j for i, j in kwargs.items() if i in self._LABEL_CONFIGS]

        # Check type and length consistencies for feature and label configs
        for i in (feature_configs, label_configs):
            types = set(map(type, i))
            sizes = set(map(len, i))
            _check_types_and_sizes(types, sizes)

        # Finally, update the configs
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

    def shape(self) -> Tuple[int, int]:
        return self.data.shape

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

    def get_feature(self, *, split_name: Optional[str] = None, return_type: FeatType = "numpy",
                    channel: Optional[str] = None, channel_type: Optional[str] = "obsm",
                    mod: Optional[str] = None):  # yapf: disable
        # Pick modality
        if mod is None:
            data = self.data
        elif not isinstance(self.data, MuData):
            raise AttributeError("`mod` option is only available when using multimodality data.")
        elif mod not in self.mod:
            raise KeyError(f"Unknown modality {mod!r}, available options are {sorted(self.mod)}")
        else:
            data = self.data.mod[mod]

        # Pick channels - obsm or varm
        channel_type = channel_type or "obsm"  # default to obsm
        if channel_type not in self._DATA_CHANNELS:
            raise ValueError(f"Unknown channel type {channel_type!r}. Available options are {self._DATA_CHANNELS}")
        channels = getattr(data, channel_type)

        # Pick feature from a specific channel
        feature = data.X if channel is None else channels[channel]

        if return_type == "default":
            if split_name is not None:
                raise ValueError(f"split_name is not supported when return_type is 'default', got {split_name=!r}")
            return feature

        # Transform features to numpy array
        if hasattr(feature, "toarray"):  # convert sparse array to dense numpy array
            feature = feature.toarray()
        elif hasattr(feature, "to_numpy"):  # convert dataframe to numpy array
            feature = feature.to_numpy()

        # Extract specific split
        if split_name is not None:
            if channel_type.startswith("obs") or channel_type == "layers":
                idx = self.get_split_idx(split_name, error_on_miss=True)
                feature = feature[idx] if channel_type == "obsm" else feature[idx][:, idx]
            elif channel_type.startswith("var"):
                logger.warning(f"Indexing option for {channel_type} not implemented yet.")

        # Convert to other data types if needed
        if return_type == "torch":
            feature = torch.from_numpy(feature)
        elif return_type != "numpy":
            raise ValueError(f"Unknown return_type {return_type!r}")

        return feature


class Data(BaseData):

    @property
    def x(self):
        return self.get_x(return_type="default")

    @property
    def y(self):
        return self.get_y(return_type="default")

    def _get(self, config_keys: List[str], *, split_name: Optional[str] = None, return_type: FeatType = "numpy",
             **kwargs) -> Any:
        info = list(map(self.config.get, config_keys))
        if all(i is None for i in info):
            mods = channels = channel_types = [None]
        else:
            mods, channels, channel_types = map(_ensure_iter, info)

        out = []
        for mod, channel, channel_type in zip(mods, channels, channel_types):
            x = self.get_feature(split_name=split_name, return_type=return_type, mod=mod, channel=channel,
                                 channel_type=channel_type, **kwargs)
            out.append(x)
        out = out[0] if len(out) == 1 else out

        return out

    def get_x(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", **kwargs) -> Any:
        """Retrieve cell features from a particular split."""
        return self._get(self._FEATURE_CONFIGS, split_name=split_name, return_type=return_type, **kwargs)

    def get_y(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", **kwargs) -> Any:
        """Retrieve cell labels from a particular split."""
        return self._get(self._LABEL_CONFIGS, split_name=split_name, return_type=return_type, **kwargs)

    def get_data(
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
        return self.get_data("train", return_type, x_kwargs, y_kwargs)

    def get_val_data(
        self, return_type: FeatType = "numpy", x_kwargs: Dict[str, Any] = dict(), y_kwargs: Dict[str, Any] = dict()
    ) -> Tuple[Any, Any]:
        """Retrieve cell features and labels from the 'val' split."""
        return self.get_data("val", return_type, x_kwargs, y_kwargs)

    def get_test_data(
        self, return_type: FeatType = "numpy", x_kwargs: Dict[str, Any] = dict(), y_kwargs: Dict[str, Any] = dict()
    ) -> Tuple[Any, Any]:
        """Retrieve cell features and labels from the 'test' split."""
        return self.get_data("test", return_type, x_kwargs, y_kwargs)
