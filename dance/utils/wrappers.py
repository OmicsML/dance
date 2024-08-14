import datetime
import functools
import time
from typing import Union

import anndata
import mudata
import numpy as np
import torch

from dance import logger
from dance.data.base import Data
from dance.typing import Any, Callable


class CastOutputType:
    """Decorator to cast the output of a function to a certain type.

    Parameters
    ----------
    target_type
        Target type to cast the output to.

    """

    def __init__(self, cast_func: Callable[[Any], Any]):
        self.cast_func = cast_func

    def __call__(self, func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            res = func(*args, **kwargs)
            typed_res = self.cast_func(res)
            return typed_res

        return wrapped_func


class TimeIt:
    """Decorator to record and show the elapsed time for a function call.

    Parameters
    ----------
    name
        Description of the function.

    """

    def __init__(self, name: str):
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


import functools


def add_mod_and_transform(cls):
    original_init = cls.__init__
    original_call = cls.__call__
    cls.add_mod_and_transform = "add_mod_and_transform"

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        mod = kwargs.pop('mod', None)
        original_init(self, *args, **kwargs)
        self.mod = mod

    @functools.wraps(original_call)
    def new_call(self, data: Data, *args, **kwargs):
        if hasattr(self, 'mod') and self.mod is not None:
            md_data = data.data
            ad_data = Data(data=transform_mod_to_anndata(md_data, self.mod))
            res = original_call(self, ad_data, *args, **kwargs)
            data.data.mod[self.mod] = ad_data.data
        else:
            return original_call(self, data, *args, **kwargs)

    cls.__init__ = new_init
    cls.__call__ = new_call
    return cls


def transform_mod_to_anndata(mod_data: mudata.MuData, mod_key: str):
    return mod_data.mod[mod_key]



