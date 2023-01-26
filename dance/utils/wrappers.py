import datetime
import functools
import time

import numpy as np
import torch

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
    """Normalize the output to a 1-d numpy array."""

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        res = func(*args, **kwargs)
        res_1d_array = np.array(res).ravel()
        return res_1d_array

    return wrapped_func


def torch_to_numpy(func):
    """Convert any torch Tensors from input arguments to numpy arrays."""

    @functools.wraps(func)
    def wrapped_func(*args):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                logger.debug("Turning torch tensor into numpy array.")
                arg = arg.detach().clone().cpu().numpy()
            new_args.append(arg)
        return func(*new_args)

    return wrapped_func
