import itertools
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from operator import is_not
from pprint import pformat

import anndata
import mudata
import numpy as np
import omegaconf
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from deprecated import deprecated

from dance import logger
from dance.typing import Any, Dict, FeatType, Iterator, List, ListConfig, Literal, Optional, Sequence, Tuple, Union


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
    elif type_ not in (list, str, ListConfig):
        raise TypeError(f"Unknownn type {type_} found in config.")


class BaseData(ABC):
    """Base data object.

    The ``dance`` data object is a wrapper of the :class:`~anndata.AnnData` object, with several utility methods to
    help retrieving data in specific splits in specific format (see :meth:`~BaseData.get_split_idx` and
    :meth:`~BaseData.get_feature`). The :class:`~anndata.AnnData` objcet is saved in the attribute ``data`` and can be
    accessed directly.

    Warning
    -------
    Since the underlying data object is a reference to the input :class:`~anndata.AnnData` object, please be extra
    cautious ***NOT*** initializing two different dance ``data`` object using the same :class:`~anndata.AnnData`
    object! If you are unsure, we recommend always initialize the dance ``data`` object using a ``copy`` of the input
    :class:`~anndata.AnnData` object, e.g.,

        >>> adata = anndata.AnnData(...)
        >>> ddata = dance.data.Data(adata.copy())

    Note
    ----
    You can directly access some main properties of :class:`~anndata.AnnData` (or :class:`~mudata.MuData` depending on
    which type of data you passed in), such as ``X``, ``obs``, ``var``, and etc.

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

    _FEATURE_CONFIGS: List[str] = ["feature_mod", "feature_channel", "feature_channel_type"]
    _LABEL_CONFIGS: List[str] = ["label_mod", "label_channel", "label_channel_type"]
    _DATA_CHANNELS: List[str] = ["obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns"]

    def __init__(self, data: Union[anndata.AnnData, mudata.MuData], train_size: Optional[int] = None, val_size: int = 0,
                 test_size: int = -1, split_index_range_dict: Optional[Dict[str, Tuple[int, int]]] = None,
                 full_split_name: Optional[str] = None):
        super().__init__()

        # Check data type
        if isinstance(data, anndata.AnnData):
            additional_channels = ["X"]
        elif isinstance(data, mudata.MuData):
            additional_channels = ["X", "mod"]
        else:
            raise TypeError(f"Unknown data type {type(data)}, must be either AnnData or MuData.")

        # Store data and pass through some main properties over
        self._data = data
        for prop in self._DATA_CHANNELS + additional_channels:
            assert not hasattr(self, prop)
            setattr(self, prop, getattr(data, prop))

        # TODO: move _split_idx_dict into data.uns
        self._split_idx_dict: Dict[str, Sequence[int]] = {}
        self._setup_splits(train_size, val_size, test_size, split_index_range_dict, full_split_name)

        if "dance_config" not in self._data.uns:
            self._data.uns["dance_config"] = dict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object that wraps (.data):\n{self.data}"

    # WARNING: need to be careful about subsampling cells as the index are not automatically updated!!
    def _setup_splits(
        self,
        train_size: Optional[Union[int, str]],
        val_size: int,
        test_size: int,
        split_index_range_dict: Optional[Dict[str, Tuple[int, int]]],
        full_split_name: Optional[str],
    ):
        if (split_index_range_dict is not None) and (full_split_name is not None):
            raise ValueError("Only one of split_index_range_dict, full_split_name can be specified, but not both")
        elif split_index_range_dict is not None:
            self._setup_splits_range(split_index_range_dict)
        elif full_split_name is not None:
            self._setup_splits_full(full_split_name)
        else:
            self._setup_splits_default(train_size, val_size, test_size)

    def _setup_splits_default(self, train_size: Optional[Union[int, str]], val_size: int, test_size: int):
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
            raise ValueError(f"Total size {tot_size:,} exceeds total number of samples {data_size:,}")

        logger.debug(f"Split sizes before conversion: {split_sizes.tolist()}")
        split_sizes[split_sizes == -1] = data_size - split_sizes.clip(0).sum()
        logger.debug(f"Split sizes after conversion: {split_sizes.tolist()}")

        split_thresholds = split_sizes.cumsum()
        for i, split_name in enumerate(split_names):
            start = split_thresholds[i - 1] if i > 0 else 0
            end = split_thresholds[i]
            if end - start > 0:  # skip empty split
                self._split_idx_dict[split_name] = list(range(start, end))

    def _setup_splits_range(self, split_index_range_dict: Dict[str, Tuple[int, int]]):
        for split_name, index_range in split_index_range_dict.items():
            if (not isinstance(index_range, tuple)) or (len(index_range) != 2):
                raise TypeError("The split index range must of a two-tuple containing the start and end index. "
                                f"Got {index_range!r} for key {split_name!r}")
            elif any(not isinstance(i, int) for i in index_range):
                raise TypeError("The split index range must of a two-tuple of int type. "
                                f"Got {index_range!r} for key {split_name!r}")

            start, end = index_range
            if end - start > 0:  # skip empty split
                self._split_idx_dict[split_name] = list(range(start, end))

    def _setup_splits_full(self, full_split_name: str):
        self._split_idx_dict[full_split_name] = list(range(self.shape[0]))

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
        """Return the dance data object configuration dict.

        Notes
        -----
        The configuration dictionary is saved in the ``data`` attribute, which is an :class:`~anndata.AnnData`
        object. Inparticular, the config will be saved in the ``.uns`` attribute with the key ``"dance_config"``.

        """
        return self._data.uns["dance_config"]

    def set_config(self, *, overwrite: bool = False, **kwargs):
        """Set dance data object configuration.

        See
        :meth: `~BaseData.set_config_from_dict`.

        """
        self.set_config_from_dict(kwargs, overwrite=overwrite)

    def set_config_from_dict(self, config_dict: Dict[str, Any], *, overwrite: bool = False):
        """Set dance data object configuration from a config dict.

        Parameters
        ----------
        config_dict
            Configuration dictionary.
        overwrite
            Used to determine the behaviour of resolving config conflicts. In the case of a conflict, where the config
            dict passed contains a key with value that differs from an existing setting, if ``overwrite`` is set to
            ``False``, then raise a ``KeyError``. Otherwise, overwrite the configuration with the new values.

        """
        # Check config key validity
        all_configs = set(self._FEATURE_CONFIGS + self._LABEL_CONFIGS)
        if (unknown_options := set(config_dict).difference(all_configs)):
            raise KeyError(f"Unknown config option(s): {unknown_options}, available options are: {all_configs}")

        feature_configs = [j for i, j in config_dict.items() if i in self._FEATURE_CONFIGS and j is not None]
        label_configs = [j for i, j in config_dict.items() if i in self._LABEL_CONFIGS and j is not None]

        # Check type and length consistencies for feature and label configs
        for i in [feature_configs, label_configs]:
            types = set(map(type, i))
            sizes = set(map(len, i))
            _check_types_and_sizes(types, sizes)

        # Finally, update the configs
        for config_key, config_val in config_dict.items():
            # New config
            if config_key not in self.config:
                if isinstance(config_val, ListConfig):
                    config_val = omegaconf.OmegaConf.to_object(config_val)
                    logger.warning(f"transform ListConfig {config_val} to List")
                self.config[config_key] = config_val
                logger.info(f"Setting config {config_key!r} to {config_val!r}")
                continue

            # Existing config
            if (old_config_val := self.config[config_key]) == config_val:  # new value is the same as before
                continue
            elif overwrite:  # new value differs from before and overwrite setting is turned on
                self.config[config_key] = config_val
                logger.warning(f"Overwriting config {config_key!r} to {config_val!r} (previously {old_config_val!r})")
            else:  # new value differs from before but overwrite setting is not on
                raise KeyError(f"Config {config_key!r} exit with value {old_config_val!r} but trying to set to a "
                               f"different value {config_val!r}. If you want to overwrite the config, please specify "
                               "`overwrite=True` when calling the set config function.")

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

    @property
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
        split_name
            Name of the split to retrieve.
        error_on_miss
            If set to True, raise KeyError if the queried split does not exit, otherwise return None.

        See Also
        --------
        :meth:`~get_split_mask`

        """
        if split_name is None:
            return list(range(self.shape[0]))
        elif split_name in self._split_idx_dict:
            return self._split_idx_dict[split_name]
        elif error_on_miss:
            raise KeyError(f"Unknown split {split_name!r}. Please set the split inddices via set_split_idx first.")
        else:
            return None

    def get_split_mask(self, split_name: str, return_type: FeatType = "numpy") -> Union[np.ndarray, torch.Tensor]:
        """Obtain mask representation of a particular split.

        Parameters
        ----------
        split_name
            Name of the split to retrieve.
        return_type
            Return numpy array if set to 'numpy', or torch Tensor if set to 'torch'.

        """
        split_idx = self.get_split_idx(split_name, error_on_miss=True)
        if return_type == "numpy":
            mask = np.zeros(self.shape[0], dtype=bool)
        elif return_type == "torch":
            mask = torch.zeros(self.shape[0], dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported return_type {return_type!r}. Available options are 'numpy' and 'torch'.")
        mask[split_idx] = True
        return mask

    def get_split_data(self, split_name: str) -> Union[anndata.AnnData, mudata.MuData]:
        """Obtain the underlying data of a particular split.

        Parameters
        ----------
        split_name
            Name of the split to retrieve.

        """
        split_idx = self.get_split_idx(split_name, error_on_miss=True)
        return self.data[split_idx]

    @staticmethod
    def _get_feature(
        in_data: Union[anndata.AnnData, mudata.MuData],
        channel: Optional[str],
        channel_type: Optional[str],
        mod: Optional[str],
    ) -> Union[np.ndarray, sp.spmatrix, pd.DataFrame]:
        # Pick modality
        if mod is None:
            data = in_data
        elif not isinstance(in_data, mudata.MuData):
            raise AttributeError("`mod` option is only available when using multimodality data.")
        elif mod not in in_data.mod:
            raise KeyError(f"Unknown modality {mod!r}, available options are {sorted(data.mod)}")
        else:
            data = in_data.mod[mod]

        if channel_type == "X":
            feature = data.X
        elif channel_type == "raw_X":
            feature = data.raw.X
        else:
            # Pick channels (obsm, varm, ...)
            channel_type = channel_type or "obsm"  # default to obsm
            if channel_type not in (options := BaseData._DATA_CHANNELS):
                raise ValueError(f"Unknown channel type {channel_type!r}. Available options are {options}")
            channel_obj = getattr(data, channel_type)

            # Pick feature from a specific channel
            if channel is None:
                # FIX: channel default change to "X".
                warnings.warn(
                    "The `None` option for channel when channel_type is no longer supported "
                    "and will raise an exception in the near future version. Please change "
                    "channel_type to 'X' to preserve the current behavior", DeprecationWarning, stacklevel=2)
                feature = data.X
            else:
                feature = channel_obj[channel]

        return feature

    def get_feature(self, *, split_name: Optional[str] = None, return_type: FeatType = "numpy",
                    channel: Optional[str] = None, channel_type: Optional[str] = "obsm",
                    mod: Optional[str] = None):  # yapf: disable
        """Retrieve features from data.

        Parameters
        ----------
        split_name
            Name of the split to retrieve. If not set, return all.
        return_type
            How should the features be returned. **sparse**: return as a sparse matrix; **numpy**: return as a numpy
            array; **torch**: return as a torch tensor; **anndata**: return as an anndata object.
        channel
            Return a particular channel as features. If ``channel_type`` is ``X`` or ``raw_X``, then return ``.X`` or
            the ``.raw.X`` attribute from the :class:`~anndata.AnnData` directly. If ``channel_type`` is ``obs``, return
            the column named by ``channel``, similarly for ``var``. Finally, if ``channel_type`` is ``obsm``, ``obsp``,
            ``varm``, ``varp``, ``layers``, or ``uns``, then return the value correspond to the ``channel`` in the
            dictionary.
        channel_type
            Channel type to use, default to ``obsm`` (will be changed to ``X`` in the near future).
        mod
            Modality to use, default to ``None``. Options other than ``None`` are only available when the underlying
            data object is :class:`~mudata.Mudata`.

        """
        feature = self._get_feature(self.data, channel, channel_type, mod)

        # FIX: no longer allow channel_type=None, use channel_type='X' or 'raw_X' instead
        channel_type = channel_type or "obsm"

        if return_type == "default":
            if split_name is not None:
                raise ValueError(f"split_name is not supported when return_type is 'default', got {split_name=!r}")
            return feature

        if return_type == "sparse":
            if isinstance(feature, np.ndarray):
                feature = sp.csr_matrix(feature)
            elif not isinstance(feature, sp.spmatrix):
                raise ValueError(f"Feature is not sparse, got {type(feature)}")
        # Transform features to numpy array
        elif hasattr(feature, "toarray"):  # convert sparse array to dense numpy array
            feature = feature.toarray()
        elif hasattr(feature, "to_numpy"):  # convert dataframe to numpy array
            feature = feature.to_numpy()

        # Extract specific split
        if split_name is not None:
            if channel_type in ["X", "raw_X", "obs", "obsm", "obsp", "layers"]:
                idx = self.get_split_idx(split_name, error_on_miss=True)
                idx = list(filter(lambda a: a < feature.shape[0], idx))
                feature = feature[idx][:, idx] if channel_type == "obsp" else feature[idx]
            else:
                logger.warning(f"Indexing option for {channel_type!r} not implemented yet.")

        # Convert to other data types if needed
        if return_type == "torch":
            feature = torch.from_numpy(feature)
        elif return_type not in ["numpy", "sparse"]:
            raise ValueError(f"Unknown return_type {return_type!r}")
        return feature

    def append(
        self,
        data,
        *,
        mode: Optional[Literal["merge", "rename", "new_split"]] = "merge",
        rename_dict: Optional[Dict[str, str]] = None,
        new_split_name: Optional[str] = None,
        label_batch: bool = False,
        **concat_kwargs,
    ):
        """Append another dance data object to the current data object.

        Parameters
        ----------
        data
            New dance data object to be added.
        mode
            How to combine the splits from the new data and the current data. (1) ``"merge"``: merge the splits from
            the data, e.g., the training indexes from both data are used as the training indexes in the new combined
            data. (2) ``"rename"``: rename the splits of the new data and add to the current split index dictionary,
            e.g., renaming 'train' to 'ref'. Requires passing the ``rename_dict``. Raise an error if the newly renamed
            key is already used in the current split index dictionary. (3) ``"new_split"``: assign the whole new data
            to a new split. Requires pssing the ``new_split_name`` that is not already used as a split name in the
            current data. (4) ``None``: do not specify split index to the newly added data.
        rename_dict
            Optional argument that is only used when ``mode="rename"``. A dictionary to map the split names in the new
            data to other names.
        new_split_name
            Optional argument that is only used when ``mode="new_split"``. Name of the split to assign to the new data.
        label_batch
            Add "batch" column to ``.obs`` when set to True.
        **concat_kwargs
            See :meth:`anndata.concat`.

        """
        offset = self.shape[0]
        new_split_idx_dict = {i: sorted(np.array(j) + offset) for i, j in data._split_idx_dict.items()}

        if mode == "merge":
            for split_name, split_idxs in self._split_idx_dict.items():
                if split_name in new_split_idx_dict:
                    split_idxs = split_idxs + new_split_idx_dict[split_name]
                new_split_idx_dict[split_name] = split_idxs
        elif mode == "rename":
            if rename_dict is None:
                raise ValueError("Mode 'rename' is selected but 'rename_dict' is not specified.")
            elif len(common_keys := set(self._split_idx_dict) & set(rename_dict.values())) > 0:
                raise ValueError(f"'rename_dict' cannot caontain split keys present in current data: {common_keys}")
            elif len(missed_keys := [i for i in data._split_idx_dict if i not in rename_dict]) > 0:
                raise KeyError(f"Missing rename mapping for keys: {missed_keys}")
            new_split_idx_dict = {rename_dict[i]: j for i, j in new_split_idx_dict.items()}
            new_split_idx_dict.update(self._split_idx_dict)
        elif mode == "new_split":
            if new_split_name is None:
                raise ValueError("Mode 'new_split' is selected but 'new_split_name' is not specified.")
            elif not isinstance(new_split_name, str):
                raise TypeError(f"'new_split_name' must be a string, got {type(new_split_name)}: {new_split_name}.")
            elif new_split_name in self._split_idx_dict:
                raise ValueError(f"{new_split_name!r} is being used in the current splits. Please pick another name.")
            new_split_idx_dict = {new_split_name: list(range(offset, offset + data.shape[0]))}
            new_split_idx_dict.update(self._split_idx_dict)
        elif mode is None:
            new_split_idx_dict = self._split_idx_dict
        else:
            raise ValueError(f"Unknown mode {mode!r}. Available options are: 'merge', 'rename', 'new_split'")

        # NOTE: Manually merging uns cause AnnData is incapable of doing so, even with uns_merge set
        new_uns = dict(data.data.uns)
        new_uns.update(dict(self.data.uns))

        if label_batch:
            if "batch" in self.data.obs.columns:
                old_batch = self.data.obs["batch"].tolist()
            else:
                old_batch = np.zeros(self.shape[0]).tolist()
            new_batch = (np.ones(data.shape[0]) * (max(old_batch) + 1)).tolist()
            batch = list(map(int, old_batch + new_batch))

        self._data = anndata.concat((self.data, data.data), **concat_kwargs)
        self._data.uns.update(new_uns)
        self._split_idx_dict = new_split_idx_dict
        if label_batch:
            self._data.obs["batch"] = pd.Series(batch, dtype="category", index=self._data.obs.index)

        return self

    def pop(self, *, split_name: str):
        # TODO: ass more option, e.g., index
        index_to_pop = self.get_split_idx(split_name, error_on_miss=True)
        index_to_preserve = sorted(set(range(self.shape[0])) - set(index_to_pop))

        oldidx_to_newidx = {j: i for i, j in enumerate(index_to_preserve)}
        new_split_idx_dict = {}
        for split_name, split_idx in self._split_idx_dict.items():
            new_split_idx = sorted(filter(partial(is_not, None), map(oldidx_to_newidx.get, split_idx)))
            if len(new_split_idx) > 0:
                new_split_idx_dict[split_name] = new_split_idx
                logger.info(f"Updating split index for {split_name!r}. {len(split_idx):,} -> {len(new_split_idx):,}")

        self._data = self._data[index_to_preserve]
        self._split_idx_dict = new_split_idx_dict

    @deprecated("out of date")
    def filter_cells(self, **kwargs):
        """Apply cell filtering using scanpy.pp.filter_cells and update splits.

        Filters the cells in `self.data` based on the provided criteria,
        similar to `scanpy.pp.filter_cells`. Crucially, this method also
        updates the internal split indices (`train_idx`, `val_idx`, etc.)
        to reflect the cells remaining after filtering.

        Parameters
        ----------
        **kwargs
            Arguments passed directly to `scanpy.pp.filter_cells`.
            Common arguments include `min_counts`, `max_counts`,
            `min_genes`, `max_genes`. Note: `inplace` is forced to `False`
            internally to get the filter mask, then applied effectively inplace.

        Returns
        -------
        self
            Returns the instance to allow method chaining.

        Raises
        ------
        NotImplementedError
            If the underlying `self.data` is not an `anndata.AnnData` object.
            Filtering `MuData` requires more careful consideration of modalities.

        """
        if not isinstance(self.data, anndata.AnnData):
            # Filtering MuData needs careful handling: filter which modality?
            # How to sync obs across modalities after filtering one?
            raise NotImplementedError("filter_cells is currently only implemented for AnnData objects. "
                                      "Filtering MuData requires specific modality handling.")

        logger.info(f"Applying filter_cells with parameters: {kwargs}")
        original_shape = self.data.shape
        original_obs_names = self.data.obs_names.copy()

        # 1. Store original obs_names for each split
        # We need the *names* of the cells in each split before filtering
        original_split_obs_names: Dict[str, pd.Index] = {}
        for split_name, split_idx in self._split_idx_dict.items():
            if split_idx is not None and len(split_idx) > 0:
                original_split_obs_names[split_name] = original_obs_names[split_idx]
            else:
                original_split_obs_names[split_name] = pd.Index([])  # Handle empty splits

        # 2. Determine which cells to keep using scanpy's logic
        # We run it with inplace=False first to get the boolean mask
        try:
            kwargs_copy = kwargs.copy()
            kwargs_copy['inplace'] = False
            cells_mask = sc.pp.filter_cells(self.data, **kwargs_copy)
        except Exception as e:
            logger.error(f"Error during sc.pp.filter_cells execution: {e}")
            raise

        num_filtered = original_shape[0] - cells_mask.sum()

        if num_filtered == 0:
            logger.info("No cells were filtered.")
            return self  # Nothing changed

        logger.info(f"Filtering out {num_filtered} cells ({original_shape[0]} -> {cells_mask.sum()}).")

        # 3. Apply the filtering to self.data
        # Slicing creates a view or copy; we make it an explicit copy
        # to ensure the underlying data is modified cleanly.
        self._data = self.data[cells_mask, :].copy()
        logger.debug(f"Data shape after filtering: {self.data.shape}")

        # 4. Update split indices
        new_obs_names = self.data.obs_names  # Names of cells *after* filtering
        # Create a fast lookup for new index positions
        new_obs_name_to_new_idx = {name: i for i, name in enumerate(new_obs_names)}

        new_split_idx_dict = {}
        total_kept_in_splits = 0
        for split_name, original_names_in_split in original_split_obs_names.items():
            # Find which names from this original split are still in the data
            kept_names_in_split = original_names_in_split[original_names_in_split.isin(new_obs_names)]

            # Get the *new* integer indices corresponding to these kept names
            new_indices = [new_obs_name_to_new_idx[name] for name in kept_names_in_split]

            if len(new_indices) > 0:
                new_split_idx_dict[split_name] = sorted(new_indices)  # Store sorted indices
                logger.debug(f"Split '{split_name}': {len(original_names_in_split)} -> {len(new_indices)} cells.")
                total_kept_in_splits += len(new_indices)
            else:
                # Keep the split name but with an empty list, or remove?
                # Keeping it might be less surprising.
                new_split_idx_dict[split_name] = []
                logger.warning(f"Split '{split_name}' is now empty after filtering.")

        # 5. Check consistency
        if total_kept_in_splits != self.data.shape[0]:
            # This might happen if some cells were not assigned to any split initially
            logger.warning(f"Total cells in updated splits ({total_kept_in_splits}) "
                           f"does not match total cells after filtering ({self.data.shape[0]}). "
                           "This may be expected if not all original cells were in a split.")

        # Update the internal dictionary
        self._split_idx_dict = new_split_idx_dict

        # Update AnnData properties accessible directly from BaseData/Data
        for prop in self._DATA_CHANNELS + ["X"]:  # Assuming AnnData here based on check above
            if hasattr(self._data, prop):
                setattr(self, prop, getattr(self._data, prop))

        logger.info("Cell filtering complete and split indices updated.")
        return self

    # --- START NEW METHOD ---
    def filter_by_mask(self, mask: Union[Sequence[bool], pd.Series, np.ndarray], update_splits: bool = True):
        """Filter cells based on a boolean mask and optionally update splits.

        Filters the cells in `self.data` using a provided boolean mask.
        If `update_splits` is True, this method also updates the internal
        split indices (`train_idx`, `val_idx`, etc.) to reflect the cells
        remaining after filtering.

        Parameters
        ----------
        mask : Union[Sequence[bool], pd.Series, np.ndarray]
            A boolean mask (list, Series, or array) with the same length as
            the current number of cells (`self.data.shape[0]`). Cells where
            the mask is True will be kept.
        update_splits : bool, optional
            Whether to update the internal split indices to align with the
            filtered data. Defaults to True. If set to False, the split
            indices will become invalid if any cells are removed.

        Returns
        -------
        self
            Returns the instance to allow method chaining.

        Raises
        ------
        ValueError
            If the mask is not boolean or has an incorrect length.
        NotImplementedError
            If the underlying `self.data` is not an `anndata.AnnData` object
            (as filtering MuData requires more careful handling).

        """
        if not isinstance(self.data, anndata.AnnData):
            raise NotImplementedError("filter_by_mask is currently only implemented for AnnData objects.")

        # --- Input Validation ---
        if not isinstance(mask, (list, tuple, pd.Series, np.ndarray)):
            raise TypeError(f"Mask must be a sequence, Series, or ndarray, got {type(mask)}")
        if len(mask) != self.data.shape[0]:
            raise ValueError(f"Mask length ({len(mask)}) must match number of cells ({self.data.shape[0]})")
        try:
            # Ensure boolean type (handles potential integer 0/1 masks)
            mask = np.asarray(mask, dtype=bool)
        except TypeError:
            raise ValueError("Mask could not be converted to boolean.")

        num_to_keep = mask.sum()
        num_to_filter = len(mask) - num_to_keep

        if num_to_filter == 0:
            logger.info("Provided mask keeps all cells. No filtering applied.")
            return self

        logger.info(f"Filtering cells based on mask: {self.data.shape[0]} -> {num_to_keep} ({num_to_filter} removed).")

        # --- Store Original State (if updating splits) ---
        original_split_obs_names: Dict[str, pd.Index] = {}
        if update_splits:
            original_obs_names = self.data.obs_names.copy()
            for split_name, split_idx in self._split_idx_dict.items():
                if split_idx is not None and len(split_idx) > 0:
                    # Check indices are valid before using them
                    if max(split_idx) >= len(original_obs_names):
                        raise IndexError(f"Invalid index found in split '{split_name}' before filtering.")
                    original_split_obs_names[split_name] = original_obs_names[split_idx]
                else:
                    original_split_obs_names[split_name] = pd.Index([])

        # --- Apply Filtering ---
        # Slicing creates a view or copy; make it an explicit copy.
        self._data = self.data[mask, :].copy()
        logger.debug(f"Data shape after filtering: {self.data.shape}")

        # --- Update Split Indices (if requested) ---
        if update_splits:
            new_obs_names = self.data.obs_names
            new_obs_name_to_new_idx = {name: i for i, name in enumerate(new_obs_names)}
            new_split_idx_dict = {}
            total_kept_in_splits = 0

            for split_name, original_names_in_split in original_split_obs_names.items():
                kept_names_in_split = original_names_in_split[original_names_in_split.isin(new_obs_names)]
                new_indices = [new_obs_name_to_new_idx[name] for name in kept_names_in_split]

                if len(new_indices) > 0:
                    new_split_idx_dict[split_name] = sorted(new_indices)
                    logger.debug(f"Split '{split_name}': {len(original_names_in_split)} -> {len(new_indices)} cells.")
                    total_kept_in_splits += len(new_indices)
                else:
                    new_split_idx_dict[split_name] = []
                    logger.warning(f"Split '{split_name}' is now empty after filtering.")

            if total_kept_in_splits != self.data.shape[0]:
                logger.warning(f"Total cells in updated splits ({total_kept_in_splits}) "
                               f"does not match total cells after filtering ({self.data.shape[0]}). "
                               "This may be expected if not all original cells were in a split.")

            self._split_idx_dict = new_split_idx_dict
            logger.info("Split indices updated.")
        else:
            logger.warning("Filtering applied, but split indices were *not* updated as requested. "
                           "Existing split indices are now likely invalid.")

        # --- Update AnnData properties accessible directly ---
        for prop in self._DATA_CHANNELS + ["X"]:
            if hasattr(self._data, prop):
                setattr(self, prop, getattr(self._data, prop))

        logger.info("Filtering by mask complete.")
        return self


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
            try:
                x = self.get_feature(split_name=split_name, return_type=return_type, mod=mod, channel=channel,
                                     channel_type=channel_type, **kwargs)
            except Exception as e:
                settings = {
                    "split_name": split_name,
                    "return_type": return_type,
                    "mod": mod,
                    "channel": channel,
                    "channel_type": channel_type,
                    "kwargs": kwargs,
                }
                raise RuntimeError(f"Failed to get features for the following settings:\n{pformat(settings)}") from e
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
