import importlib
import inspect
import itertools
import os
import sys
from copy import deepcopy
from functools import partial, reduce
from operator import mul
from pprint import pformat

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dance import logger
from dance.config import Config
from dance.exceptions import DevError
from dance.registry import REGISTRY, REGISTRY_PREFIX, Registry, resolve_from_registry
from dance.settings import CURDIR
from dance.typing import Any, Callable, ConfigLike, Dict, FileExistHandle, List, Optional, PathLike, Tuple, Union
from dance.utils import Color, default, try_import

DEFAULT_PIPELINE_TUNING_TOP_K = 3
DEFAULT_PARAMETER_TUNING_FREQ_N = 10


class Action:
    # XXX: Raise error if other keys found, unless disabled check,
    # or provide option to register other keys?
    TYPE_KEY = "type"
    DESC_KEY = "desc"
    TARGET_KEY = "target"
    SCOPE_KEY = "scope"
    PARAMS_KEY = "params"
    SKIP_FLAG = "_skip_"

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

    @property
    def skip(self) -> bool:
        """Return ``True`` if target is set to the skip flag.

        This setting affect the pipeline object. When target is set to skip flag, then
        this action will be skiped in the pipeline enumeration.

        """
        return self.target == self.SKIP_FLAG

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
        """Iterate over pipeline elements except for the skipped ones."""
        yield from (p for p in self._pipeline if not p.skip)

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
    TUNING_PARAMS_KEY = "params_to_tune"
    DEFAULT_PARAMS_KEY = "default_params"
    PELEM_INCLUDE_KEY = "include"
    PELEM_EXCLUDE_KEY = "exclude"
    PELEM_SKIP_KEY = "skippable"
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
    def candidate_names(self) -> Optional[List[str]]:
        return getattr(self, "_candidate_names", None)

    @property
    def candidate_params(self) -> Optional[List[Dict[str, Any]]]:
        return getattr(self, "_candidate_params", None)

    @property
    def wandb_config(self) -> Optional[Dict[str, Any]]:
        return self._wandb_config

    def _resolve_pelem_plan(self, idx: int) -> Optional[List[str]]:
        # NOTE: we need to use the raw config here instead of the pipeline
        # element action object, as obtained by self[idx], since that does not
        # contain the extra information about tuning settings we need, e.g.,
        # the inclusion and exlusion settings.
        pelem_config = self.config[self.PIPELINE_KEY][idx]

        # Use fixed target if available
        if pelem_config.get(self.TARGET_KEY) is not None:
            return None, None

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

        # Set skip option
        if pelem_config.get(self.PELEM_SKIP_KEY, False):
            logger.debug("Skip flag set, adding skip option.")
            filtered_candidates.add(self.SKIP_FLAG)

        return sorted(filtered_candidates), self[idx].type

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

        if self.tune_mode == "pipeline_params":  # NOTE: when running with pipeline_params, run pipeline first
            self._tune_mode = "pipeline"
            logger.info("tune mode is set to pipeline_params, tune_mode will first be converted to pipeline")

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
        self._candidate_names = [None] * pipeline_length
        if self.tune_mode == "pipeline":
            self._candidate_pipelines = [None] * pipeline_length
            for i in range(pipeline_length):
                self._default_params[i] = pipeline_config[i].get(self.DEFAULT_PARAMS_KEY)
                self._candidate_pipelines[i], self._candidate_names[i] = self._resolve_pelem_plan(i)

        elif self.tune_mode == "params":
            self._candidate_params = [None] * pipeline_length
            for i in range(pipeline_length):
                if self.DEFAULT_PARAMS_KEY in pipeline_config[i]:
                    logger.warning(f"params tuning mode ignores {self.DEFAULT_PARAMS_KEY!r}, which is "
                                   f"currently specified pipeline element #{i}:\n\t{pipeline_config[i]}")

                # Set default params (auto set key to the current target)
                if val := pipeline_config[i].get(self.PARAMS_KEY):
                    self._default_params[i] = {self[i].target: val}

                # Set tuning params
                if val := pipeline_config[i].get(self.TUNING_PARAMS_KEY):
                    self._candidate_params[i] = OmegaConf.to_container(val)
                    self._candidate_names[i] = self[i].target

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
                idx = int(i.split(f"{Pipeline.PIPELINE_KEY}.", 1)[1].split(".", 1)[0])
                logger.debug(f"Setting pipeline element {idx} to {j}")
                pipeline[idx] = j

        if pipeline is None:
            return

        # Make sure pipeline length matches
        if len(pipeline) != pipeline_length:
            raise ValueError(f"Expecting {pipeline_length} targets specifications, "
                             f"but only got {len(pipeline)}: {pipeline}")

        logger.info(f"Pipeline plan:\n{Color('green')(pformat(pipeline))}")

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
                idx, _, key = i.split(f"{Pipeline.PARAMS_KEY}.", 1)[1].split(".", 2)
                idx = int(idx)
                logger.debug(f"Setting {key!r} for pipeline element {idx} to {j}")

                if params[idx] is None:
                    params[idx] = {}
                params[idx][key] = j

        if params is None:
            return

        # Make sure pipeline length matches
        if len(params) != pipeline_length:
            raise ValueError(f"Expecting {pipeline_length} targets specifications, "
                             f"but only got {len(params)}: {params}")

        logger.info(f"Params plan:\n{Color('green')(pformat(params))}")

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
        pipeline_params: Optional[Union[Dict[str, Any], List[str]]] = None,
        params: Optional[Union[Dict[str, Any], List[Optional[Dict[str, Any]]]]] = None,
        validate: bool = True,
        strict_params_check: bool = False,
    ) -> Config:
        """Generate config based on specified pipeline and params settings.

        See more detailed info from :meth:`generate`.

        """
        # Pre-checks
        if pipeline is None and params is None and pipeline_params is None:
            raise ValueError("At least one of 'pipeline' or 'params' or 'pipeline_params' must be specified.")
        elif self.tune_mode == "pipeline":
            if pipeline is None and pipeline_params is None:
                raise ValueError("'pipeline' or 'pipeline_params' must be specified as tune mode is set to pipeline")
            elif pipeline_params is not None and pipeline is not None:
                raise ValueError("Only one of 'pipeline_params' and 'pipeline' can exist")
            elif pipeline_params is not None and pipeline is None:
                logger.info("The content in pipeline_params will be converted to pipeline")
                pipeline = pipeline_params
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
        pipeline_params: Optional[List[str]] = None,
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
        config = self.generate_config(pipeline=pipeline, params=params, pipeline_params=pipeline_params)
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
                            "skippable": True,
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
                        "_skip_",  # <- skip flag for skipping this step
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
                                "out": "feature.cell",
                            }
                            "params_to_tune": {
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
            f"{self.PIPELINE_KEY}.{i}.{n}": {
                "values": j
            }
            for i, (j, n) in enumerate(zip(self.candidate_pipelines, self.candidate_names)) if j is not None
        }
        return search_space

    def _params_search_space(self) -> Dict[str, Dict[str, Optional[Union[str, float]]]]:
        search_space = {}
        for i, (param_dict, n) in enumerate(zip(self.candidate_params, self.candidate_names)):
            if param_dict is not None:
                for key, val in param_dict.items():
                    search_space[f"{self.PARAMS_KEY}.{i}.{n}.{key}"] = val
        return search_space

    def wandb_sweep_config(self) -> Dict[str, Any]:
        if self.wandb_config is None:
            raise ValueError("wandb config not specified in the raw config.")
        return {**self.wandb_config, "parameters": self.search_space()}

    def wandb_sweep(self) -> Tuple[str, str, str]:
        wandb = try_import("wandb")

        if "wandb" not in self.config:
            raise ValueError(f"{self.config_yaml}\nMissing wandb config.")
        wandb_entity = self.config.wandb.get("entity")
        wandb_project = self.config.wandb.get("project")
        if wandb_entity is None or wandb_project is None:
            raise ValueError(f"{self.config_yaml}\nMissing either one (or both) of wandb configs "
                             f"'entity' and 'project': {wandb_entity=!r}, {wandb_project=!r}")

        sweep_config = self.wandb_sweep_config()
        logger.info(f"Sweep config:\n{pformat(sweep_config)}")
        wandb_sweep_id = wandb.sweep(sweep=sweep_config, entity=wandb_entity, project=wandb_project)
        logger.info(Color("blue")(f"\n\n\t[*] Sweep ID: {wandb_sweep_id}\n"))

        return wandb_entity, wandb_project, wandb_sweep_id

    def wandb_sweep_agent(
        self,
        function: Callable,
        *,
        sweep_id: Optional[str] = None,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        wandb = try_import("wandb")

        if sweep_id is None:
            if entity is not None or project is not None:
                raise ValueError("Cannot specify entity or project when sweep_id is not specified "
                                 "(will be inferred from config)")
            entity, project, sweep_id = self.wandb_sweep()
        else:
            entity = self.config.wandb.get("entity")
            project = self.config.wandb.get("project")

        logger.info(f"Spawning agent: {sweep_id=}, {entity=}, {project=}, {count=}")
        wandb.agent(sweep_id, function=function, entity=entity, project=project, count=count)

        return entity, project, sweep_id


def save_summary_data(entity, project, sweep_id, summary_file_path, root_path, additional_sweep_ids=None):
    """Download sweep summary data from wandb and save to file.

    The returned dataframe includes running time, results and corresponding hyperparameters, etc.

    .. code-block:: txt

        -----------------------------------------------------------------------------------------------------------
        | id         | _runtime    | _timestamp      | acc   | _step      | _wandb_runtime   | pipeline.0         |
        -----------------------------------------------------------------------------------------------------------
        | 9hwsyumy   | 42.445390   | 1706685331.70  | 0.707  | 0          | 41               | WeightedFeaturePCA |
        | sgsr5mw4   | 38.906304   | 1706685272.85  | 0.707  | 0          | 37               | WeightedFeaturePCA |
        | czvfm197   | 48.190104   | 1706685217.47  | 0.331  | 0          | 47               | CellPCA            |
        -----------------------------------------------------------------------------------------------------------

    """
    wandb = try_import("wandb")

    summary_data = []

    summary_file_path = os.path.join(root_path, summary_file_path)
    additional_sweep_ids = (additional_sweep_ids if additional_sweep_ids is not None else [])
    additional_sweep_ids.append(sweep_id)
    for sweep_id in additional_sweep_ids:
        sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
        for run in sweep.runs:
            result = dict(run.summary._json_dict).copy()
            result.update(run.config)
            result.update({"id": run.id})
            summary_data.append(flatten_dict(result))  # get result and config
    ans = pd.DataFrame(summary_data).set_index(["id"])
    ans.sort_index(axis=1, inplace=True)

    if summary_file_path is not None:
        os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
        ans.to_csv(summary_file_path)  # save file

    return ans


def flatten_dict(d, *, parent_key="", sep="_"):
    """Flatten the nested dictionary, and the parent key is the prefix of the child.

    Parameters
    ----------
    d
        The dictionary to flatten.
    parent_key
        Parent key.
    sep
        Delimiter to use to string parent keys together with the current key.

    Returns
    -------
    dict
        Flattened dictionary.

    Example
    -------
    >>> dict1 = {"a": {"x": 1, "y": {"z": 2}}, "b": 3}
    >>> flatten_dict(dict1)
    {"a_x": 1, "a_y_z": 2, "b": 3}
    >>> flatten_dict(dict1, sep=".")
    {"a.x": 1, "a.y.z": 2, "b": 3}

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(d=v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def generate_combinations_with_required_elements(elements, required_indexes):
    required_elements = [elements[required_index] for required_index in required_indexes]
    optional_elements = [x for x in elements if x not in required_elements]

    # Sort optional elements in the same order as in the `elements` list
    optional_elements.sort(key=lambda x: elements.index(x))

    # Generate all possible combinations of optional elements
    optional_combinations = []
    for i in range(1, len(optional_elements) + 1):
        optional_combinations += list(itertools.combinations(optional_elements, i))

    # Combine required elements with optional combinations to get all possible combinations
    all_combinations = []
    for optional_combination in optional_combinations:
        all_combinations.append([x for x in elements if x in required_elements or x in optional_combination])
    return all_combinations


def generate_subsets(path, tune_mode, save_directory, file_path, log_dir, required_indexes, save_config=True,
                     root_path=None):
    """Generate subsets of original pipeline plan.

    Generate multiple yamls from the original pipeline yaml. For example, given the original pipeline plan consisting
    A, B, and C, it will then generate (A, B, C), (A, B), (A, C), (B, C), (A), (B), (C) yamls. Additionally, it will
    also return the running commands and configurations of different subsets of yaml.

    Notes
    -----
    YAML can be generated automatically, but obviously still need to manually tune the parameters to avoid errors.
    When part of the process is omitted, the function parameters need to be changed, otherwise an error will be
    reported, so different YAML adjustments are required.

    Parameters
    ----------
    path
        Path to the origin pipeline plan yaml file.
    tune_mode
        Tuning mode.
    save_directory
        Directory to save the generated subset yaml files.
    file_path
        Python execution file, usually the file name is main.
    log_dir
        Directory to save the run log when executing the pipeline tuning scripts.
    required_indexes
        index in required type of origin yaml.
    root_path
        Root directory to search for the yaml and save the generated yamls.

    Returns
    --------
    command_str: str
        Command string to run, e.g., "python examples/tuning/cta_svm/main.py \
        --config_dir=config_yamls/pipeline/subset_0_  --count=4 > temp_data/1.log 2>&1 &"
    configs: list
        Configs for the generated pipeline subsets.

    """
    root_path = default(root_path, CURDIR)
    save_directory = f"{root_path}/{save_directory}"
    path = f"{root_path}/{path}"
    config = OmegaConf.load(path)
    dict_config = DictConfig(config)
    nums = dict_config[tune_mode]
    subsets = generate_combinations_with_required_elements(nums, required_indexes)
    configs = []
    command_str = "#!/bin/bash\nlog_dir=" + log_dir + "\nmkdir -p ${log_dir}\n"
    for index, subset in enumerate(subsets):
        config_copy = dict_config.copy()
        config_copy[tune_mode] = subset
        configs.append(config_copy)
        save_path = f"{save_directory}/subset_{index}_{tune_mode}_tuning_config.yaml"
        if save_config:
            OmegaConf.save(config_copy, save_path)
        count = reduce(mul, [len(p["include"]) if "include" in p else 1 for p in subset])
        config_dir = os.path.relpath(os.path.dirname(save_path), os.path.dirname(os.path.join(root_path, file_path)))
        command_str = (command_str + f"python {file_path} --config_dir={config_dir}/subset_{index}_ --count={count} " +
                       f"> {log_dir}/{index}.log 2>&1 &\n")
    return command_str, configs


def get_step3_yaml(conf_save_path="config_yamls/params/", conf_load_path="step3_default_params.yaml",
                   result_load_path="results/pipeline/best_test_acc.csv", metric="test_acc", ascending=False,
                   step2_pipeline_planer=None, required_funs=["SetConfig"], required_indexes=[sys.maxsize],
                   root_path=None):
    """Generate the configuration file of step 3 based on the results of step 2.

    Parameters
    ----------
    conf_save_path
        Directory to save the configuration file generated in step 3.
    conf_load_path
        Parameter search range of all preprocessing functions under a specific algorithm task.
    result_load_path
        Path for the result of step 2.
    metric
        Evaluation criteria.
    ascending
        The order of the results of step 2.
    top_k
        The number of steps 2 selected.
    required_funs
        Required functions in step 3.
    required_indexes
        Location of required functions in step 3.
    root_path
        root path of all paths, defaults to the directory where the script is called.

    """
    root_path = default(root_path, CURDIR)
    conf_save_path = os.path.join(root_path, conf_save_path)
    # conf_load_path = os.path.join(root_path, conf_load_path)
    result_load_path = os.path.join(root_path, result_load_path)
    conf = OmegaConf.load(conf_load_path)
    pipeline_top_k = default(step2_pipeline_planer.config.pipeline_tuning_top_k, DEFAULT_PIPELINE_TUNING_TOP_K)
    result = pd.read_csv(result_load_path).sort_values(by=metric, ascending=ascending).head(pipeline_top_k)
    columns = sorted([col for col in result.columns if col.startswith("pipeline")])
    pipeline_names = result.loc[:, columns].values
    count = 0
    for row in pipeline_names:
        pipeline = []
        row = [i for i in row]
        for x in row:
            for k in conf.pipeline:
                if k["target"] == x:
                    pipeline.append(k)
        for i, f in zip(required_indexes, required_funs):
            for k in step2_pipeline_planer.config.pipeline:
                if "target" in k and k["target"] == f:
                    pipeline.insert(i, k)
        for p1 in step2_pipeline_planer.config.pipeline:
            if "step3_frozen" in p1 and p1["step3_frozen"]:
                for p2 in pipeline:
                    if p1["type"] == p2["type"]:
                        if "params_to_tune" in p2:
                            del p2["params_to_tune"]
                        if "default_params" in p1:
                            for target, d_p in p1.default_params.items():
                                if target == p2["target"]:
                                    p2["params"] = d_p
        temp_conf = conf.copy()
        temp_conf.pipeline = pipeline
        temp_conf.wandb = step2_pipeline_planer.config.wandb
        temp_conf.wandb.method = "bayes"
        os.makedirs(os.path.dirname(conf_save_path), exist_ok=True)
        OmegaConf.save(temp_conf, f"{conf_save_path}/{count}_test_acc_params_tuning_config.yaml")
        count += 1


def run_step3(root_path, evaluate_pipeline, step2_pipeline_planer: PipelinePlaner, tune_mode="params"):
    """Run step 3 by default.

    Parameters
    ----------
    root_path
        root path of all paths, defaults to the directory where the script is called.
    evaluate_pipeline
        Evaluation function
    step2_pipeline_planer
        Pipeline_planer of step2
    tune_mode
        tune_mode can only be set to params

    """
    wandb = try_import("wandb")

    pipeline_top_k = default(step2_pipeline_planer.config.pipeline_tuning_top_k, DEFAULT_PIPELINE_TUNING_TOP_K)
    step3_k = default(step2_pipeline_planer.config.parameter_tuning_freq_n, DEFAULT_PARAMETER_TUNING_FREQ_N)
    # Skip some of the already run step3 because in pandas, when you sort columns with exactly the same values, the results are not random.
    # Instead, pandas preserves the order of the original data. So we can skip it without causing any impact.
    step3_start_k = default(step2_pipeline_planer.config.step3_start_k, 0)
    #Some sweep_ids of step3 that have already been run
    step3_sweep_ids=step2_pipeline_planer.config.step3_sweep_ids
    step3_sweep_ids=[None] * (pipeline_top_k-step3_start_k) if step3_sweep_ids is None else (step3_sweep_ids + [None] * (pipeline_top_k-step3_start_k - len(step3_sweep_ids)))
    
    for i in range(pipeline_top_k):
        if i < step3_start_k:
            continue
        try:
            pipeline_planer = PipelinePlaner.from_config_file(
                f"{root_path}/config_yamls/{tune_mode}/{i}_test_acc_{tune_mode}_tuning_config.yaml")
            entity, project, step3_sweep_id = pipeline_planer.wandb_sweep_agent(
                partial(evaluate_pipeline, tune_mode, pipeline_planer), sweep_id=step3_sweep_ids[i],
                count=step3_k)  # score can be recorded for each epoch
            save_summary_data(entity, project, step3_sweep_id, f"results/{tune_mode}/{i}_best_test_acc.csv",
                              root_path=root_path)
        except wandb.UsageError:
            logger.warning("continue")
            continue


# def get_params(preprocessing_pipeline:Pipeline,type,key,name):
#     ans=[]
#     pips=list(filter(lambda p: p.type==type, preprocessing_pipeline.config.pipeline))
#     for p in pips:
#         ans.append(p[key][name])
#     return ans
