import logging
import logging.config
import os
from pathlib import Path
from typing import Union

_logger_config = {
    "version": 1,
    "formatters": {
        "default_formatter": {
            "format": "[%(levelname)s][%(asctime)s][%(name)s][%(funcName)s] %(message)s",
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "default_formatter",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "dance": {
            "handlers": ["stream_handler"],
            "level": os.environ.get("DANCE_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}
logging.config.dictConfig(_logger_config)


def change_log_level(name: str = "dance", /, *, level: Union[str, int]):
    """Change logger level.

    Parameters
    ----------
    name: str
        Name of the logger whose log level is to be updated. Default set to global logger.
    level: Union[str, int]
        Log level to be updated. Can be either the log level name (str type) or its corresponding code (int type).

    """
    logging.getLogger(name).setLevel(level)


CURDIR = Path().resolve()
DANCEDIR = Path(__file__).resolve().parents[1]
DANCEPKGDIR = DANCEDIR / "dance"
METADIR = DANCEPKGDIR / "metadata"
ATLASDIR = DANCEDIR / "examples/atlas"
SIMILARITYDIR = ATLASDIR / "sc_similarity_examples"
entity = "xzy11632"
project = "dance-dev"
__all__ = [
    "change_log_level",
]
