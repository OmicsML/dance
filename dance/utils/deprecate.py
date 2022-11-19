import functools

from dance import logger


def deprecated(func):
    """Wrap a function that is to be deprecated with a deprecation warning message."""

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        logger.warning(f"The function {func!r} is deprecated and will be removed soon.")
        return func(*args, **kwargs)

    return wrapped_func
