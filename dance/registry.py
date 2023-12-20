from functools import partial

from dance import logger
from dance.typing import Any, Dict, Iterator, Optional, Tuple

REGISTRY_PREFIX = "_registry_"

TYPE_KEY = "type"
DESC_KEY = "desc"
TARGET_KEY = "target"
SCOPE_KEY = "scope"
PARAMS_KEY = "params"


class DotDict(dict):
    """Special dictionary equiped with dot compositional key handling.

    Example
    -------
    .. code-block:: python

        dotdict = DotDict({"a": {"b": 1}})
        assert dotdict.a.b == dotdict.get("a.b") == dotdict["a"]["b"] == 1

    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dictionary: Optional[Dict[str, Any]] = None):
        dictionary = dictionary or {}
        for key, val in dictionary.items():
            if "." in key:
                raise ValueError(f"key for DotDict cannot contain '.': {key!r}")
            self[key] = DotDict(val) if hasattr(val, "keys") else val

    def get(self, key: str, default: Any = None, missed_ok: bool = True, create_on_miss: bool = False):
        """Get value given the compositional key.

        Parameters
        ----------
        key
            A multi-level key concatenated by ``"."``. If set to an empty string ``""``, then return self.
        default
            Default value to be returned if queried entry not found.
        missed_ok
            If set to ``True`` (default), then return the default value when the queried entry does not exist.
            Otherwise, raise a ``KeyError`` in the event of failing to find queried entry.
        create_on_miss
            If set to ``True``, then create a new node (an empty :class:`DotDict`) for the queried entry. Enabling
            ``create_on_miss`` must also have ``missed_ok`` enabled as well.

        """
        if create_on_miss and not missed_ok:
            raise ValueError("missed_ok must be enabled when create_on_miss is enabled.")

        keys = key.split(".")
        if len(keys) == 1 and keys[0] == "":
            return self

        out = self
        try:
            for i in keys:
                out = out[i]
        except KeyError as e:
            if create_on_miss:
                node = DotDict()
                self.set(key, node)
                return node

            if missed_ok:
                return default

            raise KeyError(f"Failed to decode keys {keys!r}") from e

        return out

    def set(self, key: str, val: Any, exist_ok: bool = True):
        """Set value given the compositional key.

        Parameters
        ----------
        key
            A multi-level key concatenated by ``"."``. If set to an empty string ``""``, then return self.
        val
            Value to be set.
        exist_ok
            If set to ``False`` (default), then raise ``KeyError`` if the entry to be written already exist.

        """
        if not exist_ok and self.get(key) is not None:
            raise KeyError(f"Key exists: {key}")

        keys = key.split(".")
        subdict = self
        for i, j in enumerate(keys[:-1]):
            subdict = subdict.setdefault(j, DotDict())
            if not isinstance(subdict, DotDict):
                raise KeyError(f"Level {i} ({j!r}) is already set as a non-leaf node: {subdict}.")

        subdict[keys[-1]] = val


class Registry(DotDict):

    def is_leaf_node(self, key: str) -> bool:
        return not isinstance(self.get(key), DotDict)

    def children(
        self,
        key: str = "",
        leaf_node: bool = True,
        non_leaf_node: bool = True,
        return_val: bool = False,
        _level: int = 0,
    ) -> Iterator[Any]:
        """Iterate over the children nodes.

        Parameters
        ----------
        key
            Multi-level key (see :meth:`DotDict.get`) for the level to begin iterating.
        leaf_node
            Return leaf node if set to ``True`` (default).
        nonleaf_node
            Return non-leaf node if set to ``True`` (default).
        return_val
            If set to ``True``, then return the value along with the keys when iterating over the children of a
            non-leaf node. If set to ``False`` (default), then only return the keys.

        Example
        -------
        .. code-block:: python

            r = Registry({"a": 1, "b": {"c": 2}})
            assert list(r.children(leaf_node=True, non_leaf_node=True)) == ["a", "b", "b.c"]
            assert list(r.children("b", leaf_node=True, non_leaf_node=True)) == []
            assert list(r.children(leaf_node=False, non_leaf_node=True)) == ["b"]
            assert list(r.children(leaf_node=True, non_leaf_node=False)) == ["a", "b.c"]
            assert list(r.children(leaf_node=True, non_leaf_node=False, return_val=True)) == [("a", 1), ("b.c", 2)]

        """
        if not non_leaf_node and not leaf_node:
            raise ValueError("Either one, or both, of leaf_node and non_leaf_node must be True")

        if _level == 0 and self.is_leaf_node(key):
            raise KeyError(f"{key} is a leaf node. children only take non-leaf nodes.")

        def return_(k):
            return k if not return_val else (k, self.get(k))

        node = self.get(key)
        kwargs = dict(leaf_node=leaf_node, non_leaf_node=non_leaf_node, return_val=return_val)

        for i, j in node.items():
            key_ = ".".join((key, i)).lstrip(".")

            if self.is_leaf_node(key_):
                if leaf_node:
                    yield return_(key_)
            else:
                if non_leaf_node:
                    yield return_(key_)
                yield from self.children(key_, _level=_level + 1, **kwargs)


def resolve_from_registry(name: str, scope: str):
    scope = scope.replace(REGISTRY_PREFIX, "", 1).lstrip(".")


REGISTRY = Registry()


def register(*scope: Tuple[str], name: Optional[str] = None, registry: Registry = REGISTRY):

    def wrapped_obj(obj):
        logger.debug(f"Registering {obj!r}")

        # Infer name
        if name is None:
            obj_name = obj.__name__
            logger.debug(f"Inferring name of {obj!r} based on function name {name!r}")
        else:
            obj_name = name

        # Register
        try:
            key = ".".join((*scope, obj_name))
            registry.set(key, obj, exist_ok=False)
        except KeyError as e:
            raise KeyError(f"{obj_name!r} already registered under {scope}") from e

        return obj

    return wrapped_obj


register_dataset = partial(register, "dataset")
register_preprocessor = partial(register, "preprocessor")

register_genestats_func = partial(register, "function", "genestats")
register_metric_func = partial(register, "function", "metric")

REGISTERED_DATASETS = REGISTRY.get("dataset", create_on_miss=True)
REGISTERED_PREPROCESSORS = REGISTRY.get("preprocesor", create_on_miss=True)
REGISTERED_GENESTATS_FUNCS = REGISTRY.get("function.genestats", create_on_miss=True)
REGISTERED_METRIC_FUNCS = REGISTRY.get("function.metric", create_on_miss=True)

__all__ = [
    "REGISTERED_DATASETS",
    "REGISTERED_GENESTATS_FUNCS",
    "REGISTERED_METRIC_FUNCS",
    "REGISTERED_PREPROCESSORS",
    "REGISTRY",
    "register_dataset",
    "register_genestats_func",
    "register_metric_func",
    "register_preprocessor",
]
