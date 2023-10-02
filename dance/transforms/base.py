import hashlib
import warnings
from abc import ABC, abstractmethod

from anndata import AnnData

from dance import logger
from dance.data.base import Data
from dance.typing import LogLevel, Optional, Tuple


class BaseTransform(ABC):
    """BaseTransform abstract object.

    Parameters
    ----------
    log_level
        Logging level.
    out
        Name of the obsm channel or layer where the transformed features will be saved. Use the current
        transformation name if it is not set.

    """

    _DISPLAY_ATTRS: Tuple[str] = ()

    def __init__(self, out: Optional[str] = None, log_level: LogLevel = "WARNING"):
        self.out = out or self.name

        self.logger = logger.getChild(self.name)
        self.logger.setLevel(log_level)
        self.log_level = log_level

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def hexdigest(self) -> str:
        """Return MD5 hash using the representation of the transform object."""
        return hashlib.md5(repr(self).encode()).hexdigest()

    def __repr__(self) -> str:
        display_attrs_str_list = [f"{i}={getattr(self, i)!r}" for i in self._DISPLAY_ATTRS]
        display_attrs_str = ", ".join(display_attrs_str_list)
        return f"{self.name}({display_attrs_str})"

    @abstractmethod
    def __call__(self, data: Data) -> Data:
        raise NotImplementedError


class AnnDataAdaptor:
    """Adaptor for transforming AnnData instead of dance data object.

    Example
    -------
    Modify an :class:`AnnData` object inplace

        >>> AnnDataAdaptor(FilterGenes(mode="sum"))(adata)

    """

    def __init__(self, transform, **data_init_kwargs):
        warnings.warn(
            "This is a temporary patch to enable transforming AnnData, and "
            "might have potential incompatibility issues. Use cautiously.",
            UserWarning,
            stacklevel=2,
        )
        self.transform = transform
        self.data_init_kwargs = data_init_kwargs

    def __call__(self, adata: AnnData) -> AnnData:
        data = Data(adata, **self.data_init_kwargs)
        self.transform(data)
        return data.data
