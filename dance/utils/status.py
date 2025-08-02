import functools
import inspect

from dance import logger
from dance.typing import Any, Optional
from dance.utils import Color, default


def deprecated(func=None, *, msg=None, version_removed=None):
    """Wrap a function that is to be deprecated with a deprecation warning message."""

    if func is None:
        return functools.partial(deprecated, msg=msg, version_removed=version_removed)

    when = default(version_removed, "soon")
    when = "soon" if version_removed is None else f"in {version_removed}"
    reason = "" if msg is None else Color("red")(f"\n   Reason: {msg}")

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        logger.warning(f"The function {func.__name__!r} ({inspect.getabsfile(func)}) "
                       f"is deprecated and will be removed {when}{reason}")
        return func(*args, **kwargs)

    return wrapped_func


def experimental(obj: Optional[Any] = None, *, reason: Optional[str] = None, _level: int = 0):
    """Wrap a object that is under development with a warning message."""
    if obj is None:
        if _level >= 1:
            raise RuntimeError(f"Failed to obtain function to be bounded, entered dead loop: {_level=}")
        return functools.partial(experimental, reason=reason, _level=_level + 1)

    @functools.wraps(obj)
    def wrapped_func(*args, **kwargs):
        detail = "" if reason is None else f"\n   Reason: {reason}"
        logger.warning(f"{obj!r} is an experimental faeture that is currently under development. "
                       f"Changes might be made and could break compatibility in the future. Use cautiously.{detail}")
        return obj(*args, **kwargs)

    return wrapped_func
