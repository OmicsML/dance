from abc import ABC, abstractmethod

from dance import logger
from dance.typing import Any, LogLevel, Optional


class BaseTransform(ABC):

    def __init__(self, out: Optional[str] = None, log_level: LogLevel = "WARNING"):
        """Initialize transformation.

        Parameters
        ----------
        log_level
            Logging level.
        out
            Name of the obsm channel or layer where the transformed features will be saved. Use the current
            transformation name if it is not set.

        """
        self.out = out or self.name

        self.logger = logger.getChild(self.name)
        self.logger.setLevel(log_level)
        self.log_level = log_level

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def name(self) -> str:
        return self.__class__.__name__
