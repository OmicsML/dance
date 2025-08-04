import logging

from dance import settings

logger = logging.getLogger("dance")

__version__ = "1.0.1-dev"
__all__ = [
    "logger",
    "settings",
]
