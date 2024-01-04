import importlib
import inspect
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf

from dance import logger
from dance.config import Config
from dance.exceptions import DevError
from dance.registry import REGISTRY, REGISTRY_PREFIX, Registry, resolve_from_registry
from dance.typing import Any, Callable, ConfigLike, Dict, FileExistHandle, List, Optional, PathLike, Union
from dance.utils import default


class Action:
    # XXX: Raise error if other keys found, unless disabled check,
    # or provide option to register other keys?
    TYPE_KEY = "type"
    DESC_KEY = "desc"
    TARGET_KEY = "target"
    SCOPE_KEY = "scope"
    PARAMS_KEY = "params"

    def __init__(
        self,
        *,
        type_: Optional[str] = None,
        desc: Optional[str] = None,
        target: Optional[str] = None,
        scope: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        _parent_type: Optional[str] = None,
        _registry: Registry = REGISTRY,
    ):
        self._type = type_
        self._parent_type = _parent_type
        self._desc = desc  # TODO: extract default description from docstring?
        self._target = target
        self.scope = scope  # defaults to REGISTRY_PREFIX
        self._params = default(params, {})
        self._registry = _registry  # for testing purposes

    @property
    def type(self) -> Optional[str]:
        return self._type

    @property
    def parent_type(self) -> Optional[str]:
        return self._parent_type

    @property
    def full_type(self) -> Optional[str]:
        if self.type is None and self.parent_type is None:
            return None
        else:
            return ".".join(filter(None, (self.parent_type, self.type)))

    @property
    def desc(self) -> Optional[str]:
        return self._desc

    @property
    def target(self) -> Optional[str]:
        return self._target

    @property
    def scope(self) -> str:
        return self._scope

    @scope.setter
    def scope(self, val: Optional[str]):
        val = default(val, REGISTRY_PREFIX)
        if val == REGISTRY_PREFIX:
            val = ".".join(filter(None, (val, self.parent_type, self.type)))
        self._scope = val

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
    def from_config(cls, cfg: ConfigLike, **kwargs):
        return cls(
            type_=cfg.get(cls.TYPE_KEY),
            desc=cfg.get(cls.DESC_KEY),
            target=cfg.get(cls.TARGET_KEY),
            scope=cfg.get(cls.SCOPE_KEY),
            params=cfg.get(cls.PARAMS_KEY),
            **kwargs,
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
    PIPELINE_KEY = "pipeline"

    # TODO: shared configs that are parsed under sub config dicts
    def __init__(self, cfg: ConfigLike, *, _parent_type: Optional[str] = None, _registry: Registry = REGISTRY):
        super().__init__(
            type_=cfg.get(self.TYPE_KEY),
            desc=cfg.get(self.DESC_KEY),
            _parent_type=_parent_type,
            _registry=_registry,
        )

        self._pipeline: List[Action] = []
        if (sub_cfgs := cfg.get(self.PIPELINE_KEY)) is None:
            raise ValueError(f"Missing pipeline config. Please specify {self.PIPELINE_KEY!r}")

        for sub_cfg in sub_cfgs:
            if self.PARAMS_KEY in sub_cfg and self.PIPELINE_KEY in sub_cfg:
                raise KeyError(f"Cannot specify both {self.PARAMS_KEY!r} and {self.PIPELINE_KEY!r} at the same time.")

            cls = Pipeline if self.PIPELINE_KEY in sub_cfg else Action
            self._pipeline.append(cls.from_config(sub_cfg, _parent_type=self.full_type, _registry=_registry))

        # NOTE: need to set config at last as config setter might use _pipeline
        self.config = cfg

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, cfg: ConfigLike):
        self._config = Config(cfg)

    @property
    def config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()

    @property
    def config_yaml(self) -> str:
        return self.config.to_yaml()

    def __iter__(self):
        yield from self._pipeline

    def __getitem__(self, idx: int) -> Action:
        return self._pipeline[idx]

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

        # Try to resolve all functionals first before returning the composed funcitonal
        try:
            for a in self._pipeline:
                a.functional
        except KeyError as e:
            raise KeyError(f"Failed to resolve for {a}:\n   scope={a.scope}\n   type={a.type}"
                           f"\n   parent_type={a.parent_type}\n   full_type={a.full_type}") from e

        # TODO: maybe come up with some mechanism to automatically figure out
        # what the input (and output) should be?
        def bounded_functional(*args, **kwargs):
            for a in self._pipeline:
                a(*args, **kwargs)

        return bounded_functional

    @classmethod
    def from_config(cls, cfg: ConfigLike, **kwargs):
        return cls(cfg, **kwargs)

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


class PipelinePlaner(Pipeline):
    TUNE_MODE_KEY = "tune_mode"
    DEFAULT_PARAMS_KEY = "default_params"
    PELEM_INCLUDE_KEY = "include"
    PELEM_EXCLUDE_KEY = "exclude"
    WANDB_KEY = "wandb"
    VALID_TUNE_MODES = ("pipeline", "params")

    def __init__(self, cfg: ConfigLike, **kwargs):
        if self.TUNE_MODE_KEY not in cfg:
            raise ValueError(f"PipelinePlaner config must contain {self.TUNE_MODE_KEY!r}")
        super().__init__(cfg, **kwargs)

    @property
    def tune_mode(self) -> str:
        return self._tune_mode

    @property
    def base_config(self) -> Config:
        return self._base_config

    @property
    def default_params(self) -> List[Optional[ConfigLike]]:
        return self._default_params

    @property
    def candidate_pipelines(self) -> Optional[List[List[str]]]:
        return getattr(self, "_candidate_pipelines", None)

    @property
    def candidate_params(self) -> Optional[List[Dict[str, Any]]]:
        return getattr(self, "_candidate_params", None)

    @property
    def wandb_config(self) -> Optional[Dict[str, Any]]:
        return self._wandb_config

    def _resolve_pelem_plan(self, idx: int) -> Optional[List[str]]:
        # NOTE: we need to use the raw config here instaed of the pipeline
        # element action object, as obtained by self[idx], since that does not
        # contain the extra information about tuning settings we need, e.g.,
        # the inclusion and exlusion settings.
        pelem_config = self.config[self.PIPELINE_KEY][idx]

        # Use fixed target if available
        if pelem_config.get(self.TARGET_KEY) is not None:
            return None

        # Disallow setting includes and excludes at the same time
        if all(pelem_config.get(i) is not None for i in (self.PELEM_INCLUDE_KEY, self.PELEM_EXCLUDE_KEY)):
            raise ValueError(f"Cannot set {self.PELEM_INCLUDE_KEY!r} and {self.PELEM_EXCLUDE_KEY!r}"
                             f" at the same time:\n{self.config[self.PIPELINE_KEY][idx]}")

        # Obtain valid candidate target options
        scope = self[idx].full_type
        try:
            candidates = {i.replace(f"{scope}.", "", 1) for i in self._registry.children(scope, non_leaf_node=False)}
        except KeyError as e:
            raise KeyError(f"Failed to resolve candidate target scope {scope!r} from registry") from e

        # Apply filtering based on inclusions or exclusions
        includes = set(pelem_config.get(self.PELEM_INCLUDE_KEY, candidates))
        if unknown_includes := includes.difference(candidates):
            logger.warning(f"{len(unknown_includes)} (out of {len(includes)}) inclusions were "
                           f"not found in the registry under the scope {scope!r}.")

        excludes = set(pelem_config.get(self.PELEM_EXCLUDE_KEY, []))
        if unknown_excludes := excludes.difference(candidates):
            logger.warning(f"{len(unknown_excludes)} (out of {len(excludes)}) exclusions were "
                           f"not found in the registry under the scope {scope!r}.")

        filtered_candidates = candidates.intersection(includes).difference(excludes)
        logger.debug(f"\n\t{includes=}\n\t{excludes=}\n\t{candidates=}\n\t{filtered_candidates=}")
        if not filtered_candidates:
            raise ValueError("Invalid pipeline tuning plan, must have at least one valid candidate:\n"
                             f"{self.config[self.PIPELINE_KEY][idx]}\n"
                             f"All available targets under the scope {scope!r}: {candidates}")

        return sorted(filtered_candidates)

    @Pipeline.config.setter
    def config(self, cfg: ConfigLike):
        """Parse raw config and set up important properties.

        1. Set tune mode.
        2. Set base config.
        3. Set default params overwrite.
        4. Set candidate pipeline/params for the planer.

        """
        # Register full config and tune mode
        self._config = Config(cfg)
        self._tune_mode = self.config.get(self.TUNE_MODE_KEY)

        pipeline_config = self.config[self.PIPELINE_KEY]
        pipeline_length = len(pipeline_config)
        if pipeline_length < 1:
            raise ValueError("Empty pipeline.")

        # Set up base config
        base_keys = pelem_keys = (self.TYPE_KEY, self.DESC_KEY, self.TARGET_KEY)
        if self.tune_mode == "pipeline":
            # NOTE: params reserved for planing when tuning mode is ``params``
            pelem_keys = pelem_keys + (self.PARAMS_KEY, )

        base_config = {}
        for key in base_keys:
            if (val := self.config.get(key)) is not None:
                base_config[key] = val
        base_pipeline_config = []
        for sub_cfg in self.config[self.PIPELINE_KEY]:
            base_pipeline_config.append({})
            for key in pelem_keys:
                if (val := sub_cfg.get(key)) is not None:
                    base_pipeline_config[-1][key] = val
        base_config[self.PIPELINE_KEY] = base_pipeline_config
        self._base_config = Config(base_config)

        # Set up candidate plans and default params
        self._default_params = [None] * pipeline_length

        if self.tune_mode == "pipeline":
            self._candidate_pipelines = [None] * pipeline_length
            for i in range(pipeline_length):
                self._default_params[i] = pipeline_config[i].get(self.DEFAULT_PARAMS_KEY)
                self._candidate_pipelines[i] = self._resolve_pelem_plan(i)

        elif self.tune_mode == "params":
            self._candidate_params = [None] * pipeline_length
            for i in range(pipeline_length):
                self._default_params[i] = pipeline_config[i].get(self.DEFAULT_PARAMS_KEY)
                if val := self[i].params:
                    self._candidate_params[i] = val

            # Make sure targets are set
            missed_target_idx = [
                i for i, j in enumerate(self.config[self.PIPELINE_KEY]) if j.get(self.TARGET_KEY) is None
            ]
            if len(missed_target_idx) > 0:
                raise ValueError(
                    "Target must be specified for all pipeline elements when tuning mode is set to 'params'. "
                    f"Missing targets for {missed_target_idx}\nFull config:{OmegaConf.to_yaml(self.config)}")

        else:
            raise ValueError(f"Unknown tune mode {self.tune_mode!r}, supported options are {self.VALID_TUNE_MODES}")

        # Other configs
        self._wandb_config = self.config.get(self.WANDB_KEY)
        if self._wandb_config is not None:
            self._wandb_config = OmegaConf.to_container(self._wandb_config)

    @staticmethod
    def _sanitize_pipeline(
        pipeline: Optional[Union[Dict[str, Any], List[str]]],
        pipeline_length: int,
    ) -> Optional[List[str]]:
        # Convert dict type pipeline to list type
        if isinstance(pipeline, dict):
            logger.debug("Automatically converting dict format pipeline specs to list."
                         f"Default pipeline dict:\n{pipeline}\n")

            pipeline_dict = pipeline
            pipeline = [None] * pipeline_length
            for i, j in pipeline_dict.items():
                idx = int(i.split(f"{Pipeline.PIPELINE_KEY}.", 1)[1])
                logger.debug(f"Setting pipeline element {idx} to {j}")
                pipeline[idx] = j

        # Make sure pipeline length matches
        if pipeline is not None and len(pipeline) != pipeline_length:
            raise ValueError(f"Expecting {pipeline_length} targets specifications, "
                             f"but only got {len(pipeline)}: {pipeline}")

        return pipeline

    @staticmethod
    def _sanitize_params(
        params: Optional[Union[Dict[str, Any], List[Optional[Dict[str, Any]]]]],
        pipeline_length: int,
    ) -> Optional[List[Optional[Dict[str, Any]]]]:
        # Convert dict type params to list type
        if isinstance(params, dict):
            logger.debug("Automatically converting dict format params specs to list of configs. "
                         f"Default params dict:\n{params}\n")

            params_dict = params
            params = [None] * pipeline_length
            for i, j in params_dict.items():
                idx, key = i.split(f"{Pipeline.PIPELINE_KEY}.", 1)[1].split(".", 1)
                idx = int(idx)
                logger.debug(f"Setting {key!r} for pipeline element {idx} to {j}")

                if params[idx] is None:
                    params[idx] = {}
                params[idx][key] = j

        if params is not None and len(params) != pipeline_length:
            raise ValueError(f"Expecting {pipeline_length} targets specifications, "
                             f"but only got {len(params)}: {params}")

        return params

    def _validate_pipeline(self, validate: bool, pipeline: List[str], i: int):
        if not validate:
            return

        if self.candidate_pipelines[i] is None:  # use fixed target
            return

        if pipeline[i] not in self.candidate_pipelines[i]:  # invalid specified target
            raise ValueError(f"Specified target {pipeline[i]} ({i=}) not supported. "
                             f"Available options are: {self.candidate_pipelines[i]}")

    def _validate_params(
        self,
        validate: bool,
        strict_params_check: bool,
        ith_target: str,
        ith_params: Dict[str, Any],
        i: int,
    ):
        if not validate:
            return

        # Check if parameters can be found from the callable's signature
        full_scope = f"{self[i].full_type}.{ith_target}"
        try:
            obj = self._registry.get(full_scope, missed_ok=False)
        except KeyError as e:
            raise DevError(
                f"Failed to obtain {full_scope} from the registry. This should have been caught earlier.") from e

        known_keys = set(inspect.signature(obj).parameters)
        if unknown_keys := set(ith_params).difference(known_keys):
            msg = (f"{len(unknown_keys)} (out of {len(ith_params)}) unknown "
                   f"params specification for {full_scope!r} ({i=}): {unknown_keys}")
            if strict_params_check:
                raise ValueError(msg)
            # FIX: need to figure out a way to get inherited kwargs as well, e.g., ``out``...
            # else:
            #     logger.warning(msg)

    def generate_config(
        self,
        *,
        pipeline: Optional[Union[Dict[str, Any], List[str]]] = None,
        params: Optional[Union[Dict[str, Any], List[Optional[Dict[str, Any]]]]] = None,
        validate: bool = True,
        strict_params_check: bool = False,
    ) -> Config:
        """Generate config based on specified pipeline and params settings.

        See more detailed info from :meth:`generate`.

        """
        # Pre-checks
        if pipeline is None and params is None:
            raise ValueError("At least one of 'pipeline' or 'params' must be specified.")
        elif pipeline is None and self.tune_mode == "pipeline":
            raise ValueError("'pipeline' must be specified as tune mode is set to pipeline")
        elif params is None and self.tune_mode == "params":
            raise ValueError("'params' must be specified as tune mode is set to params")

        # Prepare config and pipeline size
        config = self.base_config.copy()
        pipeline_length = len(config[self.PIPELINE_KEY])

        def get_ith_pelem(i: int):
            return config[self.PIPELINE_KEY][i]

        # Obtain sanitized pipeline/params specifications
        pipeline = self._sanitize_pipeline(pipeline, pipeline_length)
        params = self._sanitize_params(params, pipeline_length)

        # Main parsing loop to parse settings for each pipeline element
        # TODO: nested pipeline support?
        for i in range(pipeline_length):
            # Parse pipeline plan
            if pipeline is not None and pipeline[i] is not None:
                self._validate_pipeline(validate, pipeline, i)
                get_ith_pelem(i)[self.TARGET_KEY] = pipeline[i]

            # Parse params plan (default -> default overwirte -> parsed overwrite)
            ith_target = get_ith_pelem(i)[self.TARGET_KEY]
            ith_params = get_ith_pelem(i).get(self.PARAMS_KEY, DictConfig({}))
            if ((self.default_params[i] is not None)
                    and (ith_target in self.default_params[i])):  # default params overwrite
                ith_params = OmegaConf.merge(ith_params, self.default_params[i][ith_target])
            if params is not None and params[i] is not None:  # parsed params overwrite
                ith_params = OmegaConf.merge(ith_params, DictConfig(params[i]))
            if ith_params:  # only update if overwritten
                self._validate_params(validate, strict_params_check, ith_target, ith_params, i)
                get_ith_pelem(i)[self.PARAMS_KEY] = ith_params

        return config

    def generate(
        self,
        *,
        pipeline: Optional[List[str]] = None,
        params: Optional[List[Optional[Dict[str, Any]]]] = None,
        **kwargs,
    ) -> Pipeline:
        """Generate pipeline based on specified pipeline and params settings.

        Combine specific pipeline/params plan with the planer's blueprint to generate a specific pipeline. Abstractly,
        the overall workflow is the following:

            1. Start with the base config.
            2. Overwrite base config with defaults params overwrite given the specified targets.
            3. Overwrite with specified params (``params`` tuning mode only).

        Parameters
        ----------
        pipeline
            Pipeline specification as a list of strings indicating the each target's name for each element in the
            pipeline.
        params
            Parameter specification as a list of config to overwrite the default config specifications. Set the element
            to ``None`` to skip the params overwriting for that particular pipelin element.
        **kwargs
            Keyword arguments for initializing the Pipeline object.

        Note
        ----
        Currently, only targets obtained from the registry can be sorted as candidates. Thus, the ``scope`` option is
        disabled and will default to ``"_registry_"``.

        Example
        -------
        .. code-block:: python

            # Suppose the planer's base config is the following
            planer.base_config = {
                "type": "preprocessor",
                "pipeline": [
                    {"type": "filter.gene"},
                    {"type": "feature.cell"},
                ],
            }

            # NOTE: the followings are not exactly right but convey the main idea of the construction
            planer.generate(pipeline=["FilterGenesMarker", "CellPCA"]).config == {
                "type": "preprocessor",
                "pipeline": [
                    {"type": "filter.gene", "target": "FilterGenesMarker"},
                    {"type": "feature.cell", "target": "CellPCA"},
                ],
            }

            planer.generate(
                pipeline=["FilterGenesMarker", "CellPCA"],
                params=[None, {"n_components": 200}]
            ).config == {
                "type": "preprocessor",
                "pipeline": [
                    {"type": "filter.gene", "target": "FilterGenesMarker"},
                    {"type": "feature.cell", "target": "CellPCA", "params": {"n_components": 200}},
                ],
            }

        """
        config = self.generate_config(pipeline=pipeline, params=params)
        return Pipeline(config, **kwargs)

    def search_space(self) -> Dict[str, Any]:
        """Search space for the planer.

        The search space is designed to be used for communications with tuning
        utilities such as ``wandb``.

        Example
        -------
        .. code-block:: python

            # Pipeline plan example
            planer = PipelinePlaner(
                {
                    "type": "preprocessor",
                    "tune_mode": "pipeline",
                    "pipeline": [
                        {
                            "type": "filter.gene",
                        },
                        {
                            "type": "feature.cell",
                            "include": [
                                "WeightedFeaturePCA",
                                "CellPCA",
                                "CellSVD",
                            ],
                        },
                    ]
                }
            )

            planer.search_space() == {
                "pipeline.0.target": {
                    "values": [
                        "FilterGenesScanpy",
                        "FilterGenesCommon",
                        "FilterGenesMatch",
                        "FilterGenesPercentile",
                        "FilterGenesTopK",
                        "FilterGenesMarker",
                        "FilterGenesRegression",
                        "FilterGenesMarkerGini",
                    ],
                },
                "pipeline.1.target": {
                    "values": [
                        "WeightedFeaturePCA",
                        "CellPCA",
                        "CellSVD",
                    ],
                },
            }

            # Params plan example
            planer = PipelinePlaner(
                {
                    "type": "preprocessor",
                    "tune_mode": "pipeline",
                    "pipeline": [
                        {
                            "type": "filter.gene",
                            "target": "FilterGenesPercentile",
                        }
                        {
                            "type": "feature.cell",
                            "target": "WeightedFeaturePCA",
                            "params": {
                                "n_components": {
                                    "values": [128, 256, 512, 1024],
                                },
                                "feat_norm_mode": {
                                    "values": [None, "standardize", "l2"],
                                },
                            }
                        },
                    ]
                }
            )

            planer.search_space() == {
                "params.1.n_components": {
                    "values": [128, 256, 512, 1024],
                },
                "params.1.feat_norm_mode": {
                    "values": ["normalize", "standardize", "l2"],
                },
            }

        """
        if self.tune_mode == "pipeline":
            search_space = self._pipeline_search_space()
        elif self.tune_mode == "params":
            search_space = self._params_search_space()
        else:
            raise DevError(f"Unknown tune mode {self.tune_mode}. This should have been caught at init.")
        return search_space

    def _pipeline_search_space(self) -> Dict[str, str]:
        search_space = {
            f"{self.PIPELINE_KEY}.{i}": {
                "values": j
            }
            for i, j in enumerate(self.candidate_pipelines) if j is not None
        }
        return search_space

    def _params_search_space(self) -> Dict[str, Dict[str, Optional[Union[str, float]]]]:
        search_space = {}
        for i, param_dict in enumerate(self.candidate_params):
            if param_dict is not None:
                for key, val in param_dict.items():
                    search_space[f"{self.PARAMS_KEY}.{i}.{key}"] = val
        return search_space

    def wandb_sweep_config(self, as_yaml: bool = False) -> Dict[str, Any]:
        if self.wandb_config is None:
            raise ValueError("wandb config not specified in the raw config.")
        return {**self.wandb_config, "parameters": self.search_space()}
