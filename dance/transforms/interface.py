from dance.transforms.base import BaseTransform
from dance.typing import Any


class AnnDataTransform(BaseTransform):
    """AnnData transformation interface object.

    This object provides an interface with any function that apply in-place transformation to an AnnData object.

    Example
    -------
    Any one of the :mod:`scanpy.pp` functions should be supported. For example, we can use the
    :func:`scanpy.pp.normalize_total` function on the dance data object as follows

    >>> AnnDataTransform(scanpy.pp.normalize_total, target_sum=10000)(data)

    where ``data`` is a dance data object, e.g., :class:`dance.data.Data`. Calling the above function is effectively
    equivalent to calling

    >>> scanpy.pp.normalize_total(data.data, target_sum=10000)

    """

    _DISPLAY_ATTRS = ("func", "func_kwargs")

    def __init__(self, func: Any, **kwargs):
        """Initialize the AnnDataTransform object.

        Parameters
        ----------
        func
            In-place AnnData transformation function, e.g., any one of the :mod:`scanpy.pp` functions.
        **kwargs
            Keyword arguments for the transformation function.

        """
        super().__init__()

        self.func = func
        self.func_kwargs = kwargs

    def __repr__(self):
        func_name = f"{self.func.__module__}.{self.func.__name__}"
        return f"{self.name}(func={func_name}, func_kwargs={self.func_kwargs})"

    def __call__(self, data):
        self.func(data.data, **self.func_kwargs)
