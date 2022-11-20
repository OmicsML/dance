from abc import ABC

from dance import logger
from dance.typing import Any, LogLevel, Optional


class BaseTransform(ABC):

    def __init__(self, out_channel: Optional[str] = None, log_level: LogLevel = "WARNING"):
        """Initialize transformation.

        Parameters
        ----------
        out_channel
            Name of the channel where the transformed features will be saved. Use the current transformation name if
            not set.

        """
        self.out_channel = out_channel or self.__class__.__name__

        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
