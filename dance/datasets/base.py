import os
import os.path as osp
import pathlib
import pickle
from abc import ABC, abstractmethod

from dance import logger
from dance.data import Data
from dance.transforms.base import BaseTransform
from dance.typing import Any, Dict, List, Optional, Tuple, Union
from dance.utils import hexdigest
from dance.utils.wrappers import TimeIt


class BaseDataset(ABC):
    """BaseDataset abstract object.

    Parameters
    ----------
    root
        Root directory of the dataset.
    full_download
        If set to ``True``, then attempt to download all raw files of the dataset.

    """

    _DISPLAY_ATTRS: Tuple[str] = ()

    def __init__(self, root: str, full_download: bool = False):
        self.root = pathlib.Path(root).resolve()
        self.full_download = full_download

    def hexdigest(self) -> str:
        """Return MD5 hash using the string valued items in __dict__."""
        hash_components = {i: j for i, j in self.__dict__.items() if isinstance(j, str)}
        md5_hash = hexdigest(str(hash_components))
        logger.debug(f"{hash_components=}, {md5_hash=}")
        return md5_hash

    def __repr__(self) -> str:
        display_attrs_str_list = [f"{i}={getattr(self, i)!r}" for i in self._DISPLAY_ATTRS]
        display_attrs_str = ", ".join(display_attrs_str_list)
        return f"{self.__class__.__name__}({display_attrs_str})"

    def download_all(self):
        """Download all raw files of the dataset."""
        raise NotImplementedError

    def is_complete_all(self) -> bool:
        """Return True if all raw files of the dataset have been downloaded."""
        raise NotImplementedError

    @abstractmethod
    def download(self):
        """Download selected files of the dataset."""
        ...

    @abstractmethod
    def is_complete(self) -> bool:
        """Return True if the selected files have been downloaded."""
        ...

    @abstractmethod
    def _load_raw_data(self) -> Any:
        ...

    @abstractmethod
    def _raw_to_dance(self, raw_data: Any, /) -> Data:
        """Convert raw data into dance data object."""
        ...

    def load_raw_data(self) -> Any:
        """Download data if necessary and return data in raw format."""
        self._maybe_download()
        raw_data = self._load_raw_data()
        return raw_data

    @TimeIt("load and process data")
    def load_data(self, transform: Optional[BaseTransform] = None, cache: bool = False,
                  redo_cache: bool = False) -> Data:
        """Load dance data object and perform transformation.

        If ``cache`` option is set, then try to load the processed data from cache. The ``cache`` file hash is supposed
        to distinguish different datasets and different transformations. In particular, it is constructed by MD5 hashing
        the concatenation of the dataset MD5 hash (see :meth:`~dance.datasets.base.BaseDataset.hexdigest`) and the
        transformation MD5 hash (:meth:`~dance.transforms.base.BaseTransform.hexdigest`). In the case of no
        transformation, i.e., ``transform=None`` the transformation MD5 hash will be the empty string ``""``.

        Parameters
        ----------
        transform
            Transformation to be applied.
        cache:
            If set to ``True``, then try to read and write cache to ``<root>/cache/<hash>.pkl``
        redo_cache:
            If set to ``True``, then redo the data loading and transformation, and overwrite the previous cache with the
            newly processed data.

        """
        cache_load = self._maybe_load_cache(transform, cache, redo_cache)
        if not isinstance(cache_load, str):
            logger.info(f"Data loaded:\n{cache_load}")
            return cache_load

        raw_data = self.load_raw_data()
        data = self._raw_to_dance(raw_data)
        logger.info(f"Raw data loaded:\n{data}")

        if transform is not None:
            if not isinstance(transform, BaseTransform):
                raise TypeError(f"transform has to be inherited from BaseTransform, got {type(transform)}: "
                                f"{transform!r}.\nIf you want to use AnnData transfomrations functions such as "
                                "scanpy.pp.log1p, please consider wrapping it with dance.transforms.AnnDataTransform")
            transform(data)
            logger.info(f"Data transformed:\n{data}")

        if cache:
            with open(cache_load, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved processed data to cache: {cache_load}")

        return data

    def _maybe_load_cache(self, transform, cache, redo_cache) -> Union[Data, str]:
        """Check and load processed data from cache if available."""
        cache_dir = osp.join(self.root, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        dataset_md5_hash = self.hexdigest()
        transform_md5_hash = "" if transform is None else transform.hexdigest()
        md5_hash = hexdigest(dataset_md5_hash + transform_md5_hash)

        cache_file_path = osp.join(cache_dir, f"{md5_hash}.pkl")
        if osp.isfile(cache_file_path) and cache:
            with open(cache_file_path, "rb") as f:
                data = pickle.load(f)
            try:
                terminal_width = os.get_terminal_size().columns
            except OSError:
                terminal_width = 80
            logger.info(f"Loading cached data at {cache_file_path}\n"
                        f"{'Cache data info':=^{terminal_width}}\n"
                        f"{'Dataset object info':-^{terminal_width}}\n{self!r}\n"
                        f"{'Transformation info':-^{terminal_width}}\n{transform!r}\n"
                        f"{'Loaded data info':-^{terminal_width}}\n{data!r}\n"
                        f"{'End of cache data info':=^{terminal_width}}")
            return data
        else:
            return cache_file_path

    def _maybe_download(self):
        """Check and download selected raw files if needed."""
        if self.full_download and not self.is_complete_all():
            logger.debug("Full download option set and not all data is available. Start downloading all...")
            self.download_all()
        elif not self.is_complete():
            logger.debug("Missing files ({self.is_complete()=!r}). Start downloading...")
            self.download()

    @classmethod
    def get_available_data(cls) -> List[Union[str, Dict[str, str]]]:
        """List available data of the dataset."""
        if hasattr(cls, "AVAILABLE_DATA"):
            return cls.AVAILABLE_DATA
        else:
            raise NotImplementedError(f"Dataset {cls.__class__.__name__} does not have AVAILABLE_DATA specified yet, "
                                      "please specify in the class definition to enable listing data availablity.")
