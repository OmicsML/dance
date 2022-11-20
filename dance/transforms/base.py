from abc import ABC

from dance.typing import Any


class BaseTransform(ABC):

    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
