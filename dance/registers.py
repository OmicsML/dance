from functools import partial

from dance import logger
from dance.typing import Any, Dict, Optional

DANCE_DATASETS: Dict[str, Any] = {}
GENESTATS_FUNCS: Dict[str, Any] = {}
METRIC_FUNCS: Dict[str, Any] = {}


def register_base(name: Optional[str], mapping: Dict[str, Any]):

    def wrapped_obj(obj):
        logger.debug(f"Registering {obj!r}")
        if name is None:
            obj_name = obj.__name__.lower()
            logger.debug(f"Inferring name of {obj!r} based on function name {name!r}")
        else:
            obj_name = name

        if obj_name in mapping:
            raise KeyError(f"Dataset {obj_name!r} already registered.")
        mapping[obj_name] = obj
        return obj

    return wrapped_obj


register_dataset = partial(register_base, mapping=DANCE_DATASETS)
register_genestats_func = partial(register_base, mapping=GENESTATS_FUNCS)
register_metric_func = partial(register_base, mapping=METRIC_FUNCS)
