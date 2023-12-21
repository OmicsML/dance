import importlib
from copy import deepcopy

from dance import logger
from dance.config import Config
from dance.registry import REGISTRY, REGISTRY_PREFIX, Registry, resolve_from_registry
from dance.typing import Any, Callable, ConfigLike, Dict, FileExistHandle, List, Optional, PathLike
from dance.utils import default


class Action:
    # XXX: Raise error if other keys found, unless disabled check,
    # or provide option to register other keys?
    TYPE_KEY = "type"
    DESC_KEY = "desc"
    TARGET_KEY = "target"
    SCOPE_KEY = "scope"
    PARAMS_KEY = "params"
    PIPELINE_KEY = "pipeline"

    def __init__(
        self,
        *,
        type_: Optional[str] = None,
        desc: Optional[str] = None,
        target: Optional[str] = None,
        scope: str = REGISTRY_PREFIX,
        params: Optional[Dict[str, Any]] = None,
        _registry: Registry = REGISTRY,
    ):
        self._type = type_
        self._desc = desc  # TODO: extract default description from docstring?
        self._target = target
        self._scope = scope
        self._params = default(params, {})
        self._registry = _registry  # for testing purposes

    @property
    def type(self) -> str:
        return self._type

    @property
    def desc(self) -> Optional[str]:
        return self._desc

    @property
    def target(self) -> str:
        return self._target

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def _get_target(self):
        if self.scope.startswith(REGISTRY_PREFIX):
            scope = self.scope
            if (self.scope == REGISTRY_PREFIX) and (self.type is not None):
                logger.debug(f"Automatically appending type {self.type} as the registry scope")
                scope = ".".join((scope, self.type))

            logger.debug(f"Searching for {self.target} from registry: {scope}")
            target = resolve_from_registry(self.target, scope, registry=self._registry)
        else:
            logger.debug(f"Searching for {self.target} in the wild: {self.scope}")
            mod = importlib.import_module(self.scope)
            target = getattr(mod, self.target)
        return target

    @property
    def functional(self) -> Callable:
        # XXX: maybe save the target so that don't need to resolve again?
        # (need to track if target/params changed)
        func_cls = self._get_target()
        return func_cls(**self.params)

    def __call__(self, *args, **kwargs):
        return self.functional(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({default(self.target, '')})"

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_config(cls, cfg: ConfigLike):
        return cls(
            type_=cfg.get(cls.TYPE_KEY),
            desc=cfg.get(cls.DESC_KEY),
            target=cfg.get(cls.TARGET_KEY),
            scope=cfg.get(cls.SCOPE_KEY),
            params=cfg.get(cls.PARAMS_KEY),
        )

    def to_config(self) -> Config:
        return Config({
            self.TYPE_KEY: self.type,
            self.DESC_KEY: self.desc,
            self.TARGET_KEY: self.target,
            self.SCOPE_KEY: self.scope,
            self.PARAMS_KEY: self.params,
        })

    def to_dict(self) -> Dict[str, Any]:
        return self.to_config().to_dict()

    def to_yaml(self) -> str:
        return self.to_config().to_yaml()

    def dump_json(self, path: PathLike, exist_handle: FileExistHandle = "warn"):
        self.to_config().dump_json(path, exist_handle)

    def dump_yaml(self, path: PathLike, exist_handle: FileExistHandle = "warn"):
        self.to_config().dump_yaml(path, exist_handle)


class Pipeline(Action):

    # TODO: shared configs that are parsed under sub config dicts
    def __init__(self, cfg: ConfigLike):
        self._config = Config(cfg)

        self._pipeline: List[Action] = []
        if (sub_cfgs := cfg.get(self.PIPELINE_KEY)) is None:
            raise ValueError(f"Missing pipeline config. Please specify {self.PIPELINE_KEY!r}")

        for sub_cfg in sub_cfgs:
            if self.PARAMS_KEY in sub_cfg and self.PIPELINE_KEY in sub_cfg:
                raise KeyError(f"Cannot specify both {self.PARAMS_KEY!r} and {self.PIPELINE_KEY!r} at the same time.")

            cls = Pipeline if self.PIPELINE_KEY in sub_cfg else Action
            self._pipeline.append(cls.from_config(sub_cfg))

        super().__init__(type_=cfg.get(self.TYPE_KEY), desc=cfg.get(self.DESC_KEY))

    @property
    def config(self) -> Config:
        self._config

    @property
    def config_dict(self) -> Dict[str, Any]:
        self.config.to_dict()

    @property
    def config_yaml(self) -> str:
        self.config.to_yaml()

    def __iter__(self):
        yield from self._pipeline

    def __len__(self):
        return len(self._pipeline)

    def __repr__(self) -> str:
        cr = "\n"
        indent = "    "
        sep = cr + indent

        # Replace carrige return with separator to take care of nested repr
        reprs = [repr(i).replace(cr, sep) for i in self]
        pipeline_str = sep.join(reprs)

        return f"{self.__class__.__name__}({sep}{pipeline_str}\n)"

    @property
    def functional(self) -> Callable:

        # TODO: maybe come up with some mechanism to automatically figure out
        # what the input (and output) should be?
        def bounded_functional(*args, **kwargs):
            for a in self._pipeline:
                a(*args, **kwargs)

        return bounded_functional

    @classmethod
    def from_config(cls, cfg: ConfigLike):
        return cls(cfg)

    @classmethod
    def from_config_file(cls, path: PathLike, **kwargs):
        cfg = Config.from_file(path, **kwargs)
        return cls.from_config(cfg)

    def to_config(self) -> Config:
        return Config({
            self.TYPE_KEY: self.type,
            self.DESC_KEY: self.desc,
            self.PIPELINE_KEY: [a.to_config() for a in self],
        })
