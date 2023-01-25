import datetime
import functools
import time

import numpy as np

from dance import logger


class TimeIt:
    """Decorator to record and show the elapsed time for a function call."""

    def __init__(self, name: str):
        """Initialize TimeIt decorator.

        Parameters
        ----------
        name
            Description of the function.

        """
        self.name = name

    def __call__(self, func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            t_start = time.perf_counter()
            res = func(*args, **kwargs)
            elapsed = time.perf_counter() - t_start
            logger.info(f"Took {datetime.timedelta(seconds=elapsed)} to {self.name}.")
            return res

        return wrapped_func


def as_1d_array(func):

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        res = func(*args, **kwargs)
        res_1d_array = np.array(res).ravel()
        return res_1d_array

    return wrapped_func
