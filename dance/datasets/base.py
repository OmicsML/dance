from abc import ABC, abstractmethod

from dance import logger
from dance.data import Data
from dance.transforms.base import BaseTransform
from dance.typing import Any, Dict, Optional

DANCE_DATASETS: Dict[str, Any] = {}


def register_dataset(name: str):

    def wrapped_obj(obj):
        if name in DANCE_DATASETS:
            raise KeyError(f"Dataset {name!r} already registered.")
        DANCE_DATASETS[name] = obj
        return obj

    return wrapped_obj


class BaseDataset(ABC):

    def __init__(self, root: str, full_download: bool = False, *args, **kwargs):
        self.root = root
        self.full_download = full_download

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

    def load_data(self, transform: Optional[BaseTransform] = None, cache: bool = False,
                  redo_cache: bool = False) -> Data:
        # TODO: try to save and load cache if cache=True
        # TODO: disregard old cache if redo_cache=True
        if (data := self._maybe_load_cache()) is not None:
            return data

        raw_data = self.load_raw_data()
        data = self._raw_to_dance(raw_data)

        if transform is not None:
            if not isinstance(transform, BaseTransform):
                raise TypeError(f"transform has to be inherited from BaseTransform, got {type(transform)}: "
                                f"{transform!r}.\nIf you want to use AnnData transfomrations functions such as "
                                "scanpy.pp.log1p, please consider wrapping it with dance.transforms.AnnDataTransform")
            transform(data)

        return data

    def _maybe_load_cache(self) -> Optional[Data]:
        """Check and load processed data from cache if available."""
        # Check if cache dir exist, return directly if not and make one
        # Hash using the key params of the dataset and transform
        # Check hash
        # TODO: hash dataset obj
        # TODO: hash transform obj
        # TODO: show loaded cache info, e.g. path to raw, transform repr, date processed
        pass

    def _maybe_download(self):
        """Check and download selected raw files if needed."""
        if self.full_download and not self.is_complete_all():
            logger.debug("Full download option set and not all data is available. Start downloading all...")
            self.download_all()
        elif not self.is_complete():
            logger.debug("Missing files ({self.is_complete()=!r}). Start downloading...")
            self.download()
