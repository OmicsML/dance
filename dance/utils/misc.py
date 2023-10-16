import argparse
import warnings
from functools import partial, wraps
from pprint import pformat
from typing import get_args

import torch

from dance import logger
from dance.typing import Callable, LogLevel, Optional
from dance.utils import get_device, set_seed


def default_parser_processor(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    setup_seed: bool = True,
    default_seed: int = 42,
    setup_loglevel: bool = True,
    setup_torch_threads: bool = False,
    default_torch_threads: int = 4,
    setup_device: bool = True,
    default_device: str = "auto",
    setup_cache: bool = True,
    print_args: bool = True,
):
    if func is None:
        return partial(
            default_parser_processor,
            name=name,
            setup_seed=setup_seed,
            default_seed=default_seed,
            setup_loglevel=setup_loglevel,
            setup_torch_threads=setup_torch_threads,
            default_torch_threads=default_torch_threads,
            setup_device=setup_device,
            default_device=default_device,
            setup_cache=setup_cache,
            print_args=print_args,
        )

    def maybe_add_arg(parser, enable, callback, callbacks, *args, **kwargs):
        if enable:
            try:
                parser.add_argument(*args, **kwargs)
            except argparse.ArgumentError:
                warnings.warn(
                    f"Argument {args[0]} already exists in parser. "
                    f"To resolve, remove {args[0]} from the wrapped parser.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            callbacks.append(callback)

    def set_seed_callback(args):
        set_seed(args.seed)

    def set_loglevel_callback(args):
        logger.setLevel(args.log_level)

    def set_torch_threads_callback(args):
        torch.set_num_threads(args.torch_threads)

    def set_device_callback(args):
        args.device = get_device(args.device)

    def set_null_callback(args):
        ...

    @wraps(func)
    def wrapped_parser():
        if name is not None:
            print(f"Start runing {name}")

        parser = func()
        callbacks = []

        maybe_add_arg(parser, setup_loglevel, set_loglevel_callback, callbacks, "--log_level", type=str, default="INFO",
                      choices=get_args(LogLevel), help="Log level.")
        maybe_add_arg(parser, setup_seed, set_seed_callback, callbacks, "--seed", type=int, default=default_seed,
                      help="Random seed.")
        maybe_add_arg(parser, setup_seed, set_torch_threads_callback, callbacks, "--torch_threads", type=int,
                      default=default_torch_threads, help="Number of threads for PyTorch.")
        maybe_add_arg(parser, setup_seed, set_device_callback, callbacks, "--device", type=str, default="auto",
                      help="Device to use. 'auto' will be replaced with auto selected device.")
        maybe_add_arg(parser, setup_seed, set_null_callback, callbacks, "--cache", action="store_true",
                      help="Cache processed data.")

        args = parser.parse_args()

        for callback in callbacks:
            callback(args)

        if print_args:
            logger.info(f"Parsed arguments:\n{pformat(vars(args))}")

        return args

    return wrapped_parser
