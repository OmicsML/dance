import itertools
from functools import partial

import pytest

from dance.pipeline import Action, Pipeline, PipelinePlaner, generate_subsets
from dance.registry import Registry


def test_action(subtests):
    with subtests.test("Empty"):
        a = Action()
        assert repr(a) == "Action()"

    with subtests.test("Absolute scope"):
        a = Action(target="BaseTransform", scope="dance.transforms.base")
        assert repr(a) == "Action(BaseTransform)"

    with subtests.test("Registry scope"):
        r = Registry()
        r.set("a.b.integer", int)

        def pass_subtest(action):
            assert repr(action) == "Action(integer)"
            assert action.functional == 0  # int() == 0

        # Full registry scope
        pass_subtest(Action(target="integer", scope="_registry_.a.b", _registry=r))

        # Auto-resolve based on type
        pass_subtest(Action(type_="a.b", target="integer", scope="_registry_", _registry=r))
        pass_subtest(Action(type_="a.b", target="integer", _registry=r))  # default scope is _registry_


def test_pipeline(subtests):
    with subtests.test("Single level"):
        p = Pipeline({"pipeline": [{"target": "BaseTransform", "scope": "dance.transforms.base"}]})
        assert repr(p) == "Pipeline(\n    Action(BaseTransform)\n)"

    with subtests.test("Multi level"):
        p = Pipeline({
            "pipeline": [
                {
                    "target": "BaseTransform",
                    "scope": "dance.transforms.base"
                },
                {
                    "pipeline": [{
                        "target": "BaseTransform",
                        "scope": "dance.transforms.base"
                    }]
                },
            ],
        })
        assert repr(p) == ("Pipeline(\n    Action(BaseTransform)\n"
                           "    Pipeline(\n        Action(BaseTransform)\n    )\n)")


def test_pipeline_scope_resolve(subtests):
    r = Registry()
    r.set("a.b.integer", int)
    r.set("a.b.c.integer2", int)

    def pass_subtest(pipeline):
        assert repr(pipeline) == "Pipeline(\n    Action(integer)\n)"
        pipeline.functional  # test resolvability

    with subtests.test("_registry_.xxx.xxx"):
        p = Pipeline({"pipeline": [{"target": "integer", "scope": "_registry_.a.b"}]}, _registry=r)
        assert repr(p) == "Pipeline(\n    Action(integer)\n)"

    with subtests.test("_registry_"):
        # Full scope in parent pipeline
        cfg = {
            "type": "a.b",
            "pipeline": [
                {
                    "target": "integer",
                    "scope": "_registry_",
                },
            ],
        }
        pass_subtest(Pipeline(cfg, _registry=r))

        # Full scope in the child pipeline
        cfg = {
            "pipeline": [
                {
                    "type": "a.b",
                    "target": "integer",
                    "scope": "_registry_",
                },
            ],
        }
        pass_subtest(Pipeline(cfg, _registry=r))

        # Full scope constructed from parent and the child
        cfg = {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "integer",
                    "scope": "_registry_",
                },
            ],
        }
        pass_subtest(Pipeline(cfg, _registry=r))

        # Default scope is _registry_
        cfg = {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "integer",
                },
            ],
        }
        pass_subtest(Pipeline(cfg, _registry=r))

        # Invalid scope a.c
        with pytest.raises(KeyError):
            cfg = {
                "type": "a",
                "pipeline": [
                    {
                        "type": "c",
                        "target": "integer",
                    },
                ],
            }
            pass_subtest(Pipeline(cfg, _registry=r))

        # Nested pipeline scope
        cfg = {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "pipeline": [{
                        "type": "c",
                        "target": "integer2",
                        "scope": "_registry_",
                    }],
                },
            ],
        }
        p = Pipeline(cfg, _registry=r)
        assert repr(p) == "Pipeline(\n    Pipeline(\n        Action(integer2)\n    )\n)"
        p.functional  # test resolvability


@pytest.fixture
def planer_toy_registry():

    def func(name, **kwargs):
        return name + "+" + "+".join(f"{i}:{j}" for i, j in kwargs.items()) if kwargs else name

    # sorted(r).children() == [
    #     "a.b.func_b0",
    #     "a.b.func_b1",
    #     "a.b.func_b2",
    #     "a.c.func_c0",
    #     "a.c.func_c1",
    #     "a.c.func_c2",
    # ]
    r = Registry()
    for i, j in itertools.product(["b", "c"], range(3)):
        name = "".join((i, str(j)))
        r.set(f"a.{i}.func_{name}", partial(func, name))

    return r


def test_pipeline_planer_construction(subtests, planer_toy_registry):
    r = planer_toy_registry

    with subtests.test("Case 0", tune_mode="pipeline"):
        # Use all availabilities from the registry
        cfg = {
            "type": "a",
            "tune_mode": "pipeline",
            "pipeline": [
                {
                    "type": "b"
                },
                {
                    "type": "c"
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b"
                },
                {
                    "type": "c"
                },
            ],
        }
        assert p.search_space() == {
            "pipeline.0.b": {
                "values": ["func_b0", "func_b1", "func_b2"]
            },
            "pipeline.1.c": {
                "values": ["func_c0", "func_c1", "func_c2"]
            },
        }

    with subtests.test("Case 1", tune_mode="pipeline"):
        # Specify inclusion
        cfg = {
            "type": "a",
            "tune_mode": "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "include": ["func_b1", "func_b2"]
                },
                {
                    "type": "c",
                    "include": ["func_c1"]
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b"
                },
                {
                    "type": "c"
                },
            ],
        }
        assert p.search_space() == {
            "pipeline.0.b": {
                "values": ["func_b1", "func_b2"]
            },
            "pipeline.1.c": {
                "values": ["func_c1"]
            },
        }

    with subtests.test("Case 1", tune_mode="pipeline", skippable=True):
        # Specify inclusion
        cfg = {
            "type":
            "a",
            "tune_mode":
            "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "include": ["func_b1", "func_b2"],
                    "skippable": True,
                },
                {
                    "type": "c",
                    "include": ["func_c1"],
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b"
                },
                {
                    "type": "c"
                },
            ],
        }
        assert p.search_space() == {
            "pipeline.0.b": {
                # NOTE: we use sorted here in case we want to change SKIP_FLAG key in the future
                "values": sorted(["func_b1", "func_b2", PipelinePlaner.SKIP_FLAG])
            },
            "pipeline.1.c": {
                "values": ["func_c1"]
            },
        }

        # NOTE: The first pelem was skipped as configured. We check it via string matching for simplicity.
        assert repr(list(p.generate(pipeline=[PipelinePlaner.SKIP_FLAG, "func_c1"]))) == "[Action(func_c1)]"

    with subtests.test("Case 2", tune_mode="pipeline"):
        # Specify exlusion
        cfg = {
            "type": "a",
            "tune_mode": "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "exclude": ["func_b1", "func_b2"]
                },
                {
                    "type": "c",
                    "exclude": ["func_c1"]
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b"
                },
                {
                    "type": "c"
                },
            ],
        }
        assert p.search_space() == {
            "pipeline.0.b": {
                "values": ["func_b0"]
            },
            "pipeline.1.c": {
                "values": ["func_c0", "func_c2"]
            },
        }

    with subtests.test("Case 3", tune_mode="pipeline"):
        # Cannot specify inclusiona nd exclusion at the same time
        cfg = {
            "type":
            "a",
            "tune_mode":
            "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "include": ["func_b0"],
                    "exclude": ["func_b1", "func_b2"]
                },
                {
                    "type": "c",
                    "exclude": ["func_c1"]
                },
            ],
        }
        with pytest.raises(ValueError):
            p = PipelinePlaner(cfg, _registry=r)

    with subtests.test("Case 4", tune_mode="pipeline"):
        # Unknown inclusion will be ignored
        cfg = {
            "type": "a",
            "tune_mode": "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "include": ["func_b1", "func_b2", "func_b3"]
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

    with subtests.test("Case 5", tune_mode="pipeline"):
        # Unknown exclusion will be ignored
        cfg = {
            "type": "a",
            "tune_mode": "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "exclude": ["func_b1", "func_b2", "func_b3"]
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {"type": "a", "pipeline": [{"type": "b"}]}
        assert p.search_space() == {"pipeline.0.b": {"values": ["func_b0"]}}

    with subtests.test("Case 0", tune_mode="params"):
        # Simple multiple choice params plan for only the first element
        cfg = {
            "type":
            "a",
            "tune_mode":
            "params",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params_to_tune": {
                        "x": {
                            "values": ["x1", "x2", "x3"]
                        },
                        "y": {
                            "values": ["y1", "y2", "y3"]
                        },
                    },
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1"
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
            ]
        }
        assert p.search_space() == {
            "params.0.func_b1.x": {
                "values": ["x1", "x2", "x3"]
            },
            "params.0.func_b1.y": {
                "values": ["y1", "y2", "y3"]
            },
        }

    with subtests.test("Case 1", tune_mode="params"):
        # Mixed params plan and repeated target in the pipeline
        cfg = {
            "type":
            "a",
            "tune_mode":
            "params",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params_to_tune": {
                        "x": {
                            "values": ["x1", "x2", "x3"]
                        },
                        "y": {
                            "values": ["y1", "y2", "y3"]
                        },
                    },
                },
                {
                    "type": "c",
                    "target": "func_c1",
                    "params_to_tune": {
                        "z": {
                            "min": 0,
                            "max": 1
                        },
                    },
                },
                {
                    "type": "c",
                    "target": "func_c1",
                    "params_to_tune": {
                        "z": {
                            "min": -10.,
                            "max": 10.
                        },
                    },
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type":
            "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1"
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
            ]
        }
        assert p.search_space() == {
            "params.0.func_b1.x": {
                "values": ["x1", "x2", "x3"]
            },
            "params.0.func_b1.y": {
                "values": ["y1", "y2", "y3"]
            },
            "params.1.func_c1.z": {
                "min": 0,
                "max": 1
            },
            "params.2.func_c1.z": {
                "min": -10.,
                "max": 10.
            },
        }

    with subtests.test("Unknown mode"):
        cfg = {
            "type": "a",
            "tune_mode": "unkown",  # unknown tune mode raises ValueError
            "pipeline": [
                {
                    "type": "b",
                    "include": ["func_b1", "func_b2"]
                },
                {
                    "type": "c",
                    "include": ["func_c1"]
                },
            ],
        }
        with pytest.raises(ValueError):
            p = PipelinePlaner(cfg, _registry=r)


def test_pipeline_planer_generation(subtests, planer_toy_registry):
    r = planer_toy_registry

    with subtests.test("Pipeline"):
        cfg = {
            "type": "a",
            "tune_mode": "pipeline",
            "pipeline": [
                {
                    "type": "b"
                },
                {
                    "type": "c"
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        with pytest.raises(ValueError):
            # Must specify config
            p.generate_config()

        with pytest.raises(ValueError):
            # Must specify pipeline config
            p.generate_config(params=["func_b1", None])

        with pytest.raises(ValueError):
            # Invalid pipeline length
            p.generate_config(pipeline=["func_b1", "func_c1", None])

        assert dict(p.generate_config(pipeline=["func_b1", "func_c1"])) == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1"
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
            ],
        }

        with pytest.raises(ValueError):
            # Unknown target for the second pipeline element
            p.generate_config(pipeline=["func_b1", "func_c100"])

        # Overwirte unkown pipeline element error
        assert dict(p.generate_config(pipeline=["func_b1", "func_c100"], validate=False)) == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1"
                },
                {
                    "type": "c",
                    "target": "func_c100"
                },
            ],
        }

    with subtests.test("Params"):
        cfg = {
            "type":
            "a",
            "tune_mode":
            "params",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params_to_tune": {
                        "x": {
                            "values": ["x1", "x2", "x3"]
                        },
                        "y": {
                            "values": ["y1", "y2", "y3"]
                        },
                    },
                },
                {
                    "type": "c",
                    "target": "func_c1",
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        with pytest.raises(ValueError):
            # Must specify config
            p.generate_config()

        with pytest.raises(ValueError):
            # Must specify params config
            p.generate_config(pipeline=[None, None])

        with pytest.raises(ValueError):
            # Invalid params length
            p.generate_config(params=[None, None, None])

        assert dict(p.generate_config(params=[None, None])) == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1"
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
            ]
        }

        ans = {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params": {
                        "x": "x1"
                    }
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
            ],
        }
        # Option 1: list of param dict
        assert dict(p.generate_config(params=[{"x": "x1"}, None])) == ans
        # Option 2: wandb type config
        assert dict(p.generate_config(params={"params.0.func_b1.x": "x1"})) == ans

        with pytest.raises(ValueError):
            # Unknown param key 'y'
            p.generate_config(params=[{"z": "z1"}, None], strict_params_check=True)

        with pytest.raises(ValueError):
            # Must specify targets
            PipelinePlaner(
                {
                    "type":
                    "a",
                    "tune_mode":
                    "params",
                    "pipeline": [{
                        "type": "b",
                        # "target": "func_b1",  # this must be set
                        "params_to_tune": {
                            "x": {
                                "values": ["x1", "x2", "x3"]
                            },
                        },
                    }],
                },
                _registry=r)

    with subtests.test("Pipeline with defaults"):
        cfg = {
            "type":
            "a",
            "tune_mode":
            "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "default_params": {
                        "func_b1": {
                            "x": "b1"
                        },
                        "func_b2": {
                            "x": "b2"
                        },
                    },
                },
                {
                    "type": "c"
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b"
                },
                {
                    "type": "c"
                },
            ],
        }
        assert p.default_params == [
            {
                "func_b1": {
                    "x": "b1"
                },
                "func_b2": {
                    "x": "b2"
                },
            },
            None,
        ]

        for i in (1, 2):
            assert dict(p.generate_config(pipeline=[f"func_b{i}", "func_c1"])) == {
                "type":
                "a",
                "pipeline": [
                    {
                        "type": "b",
                        "target": f"func_b{i}",
                        "params": {
                            "x": f"b{i}"
                        },
                    },
                    {
                        "type": "c",
                        "target": "func_c1",
                    },
                ],
            }

    with subtests.test("Params with defaults"):
        cfg = {
            "type":
            "a",
            "tune_mode":
            "params",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params": {
                        "x": "b1"
                    },
                },
                {
                    "type": "c",
                    "target": "func_c1",
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type": "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1"
                },
                {
                    "type": "c",
                    "target": "func_c1"
                },
            ],
        }
        assert p.default_params == [
            {
                "func_b1": {
                    "x": "b1"
                },
            },
            None,
        ]

        assert dict(p.generate_config(params=[None, None])) == {
            "type":
            "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params": {
                        "x": "b1"
                    },  # <- default
                },
                {
                    "type": "c",
                    "target": "func_c1",
                },
            ]
        }

        assert dict(p.generate_config(params=[{
            "x": "b1_new"
        }, None])) == {
            "type":
            "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params": {
                        "x": "b1_new"
                    },  # <- x overwritten by b1_new
                },
                {
                    "type": "c",
                    "target": "func_c1",
                },
            ]
        }

        assert dict(p.generate_config(params=[{
            "y": "b1"
        }, {
            "z": "c1"
        }])) == {
            "type":
            "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params": {
                        "x": "b1",
                        "y": "b1"
                    },  # <- y added
                },
                {
                    "type": "c",
                    "target": "func_c1",
                    "params": {
                        "z": "c1"
                    }  # <- z added
                },
            ]
        }

    with subtests.test("Pipeline with defaults and default pipeline settings"):
        cfg = {
            "type":
            "a",
            "tune_mode":
            "pipeline",
            "pipeline": [
                {
                    "type": "b",
                    "params": {
                        "shared_i": "i",
                        "shared_j": "j",
                        "shared_k": "k",
                    },
                    "default_params": {
                        "func_b1": {
                            "x": "b1",
                            "shared_j": "j_new_1",
                        },
                        "func_b2": {
                            "x": "b2",
                            "shared_j": "j_new_2",
                        },
                    },
                },
                {
                    "type": "c",
                    "target": "func_c1",  # <- fixed target (and params below)
                    "params": {
                        "y": "c1",
                    },
                },
            ],
        }
        p = PipelinePlaner(cfg, _registry=r)

        assert p.base_config == {
            "type":
            "a",
            "pipeline": [
                {
                    "type": "b",
                    "params": {
                        "shared_i": "i",
                        "shared_j": "j",
                        "shared_k": "k",
                    },
                },
                {
                    "type": "c",
                    "target": "func_c1",
                    "params": {
                        "y": "c1",
                    },
                },
            ],
        }
        assert p.default_params == [
            {
                "func_b1": {
                    "x": "b1",
                    "shared_j": "j_new_1",
                },
                "func_b2": {
                    "x": "b2",
                    "shared_j": "j_new_2",
                },
            },
            None,
        ]
        assert p.search_space() == {
            "pipeline.0.b": {
                "values": ["func_b0", "func_b1", "func_b2"]
            },
        }

        for i in (1, 2):
            assert dict(p.generate_config(pipeline=[f"func_b{i}", None])) == {
                "type":
                "a",
                "pipeline": [
                    {
                        "type": "b",
                        "target": f"func_b{i}",
                        "params": {
                            "x": f"b{i}",
                            "shared_i": "i",
                            "shared_j": f"j_new_{i}",
                            "shared_k": "k",
                        },
                    },
                    {
                        "type": "c",
                        "target": "func_c1",
                        "params": {
                            "y": "c1",
                        }
                    },
                ],
            }

        # HACK: overwrite fixed elements.
        assert dict(p.generate_config(pipeline=["func_b1", "func_c2"])) == {
            "type":
            "a",
            "pipeline": [
                {
                    "type": "b",
                    "target": "func_b1",
                    "params": {
                        "x": "b1",
                        "shared_i": "i",
                        "shared_j": "j_new_1",
                        "shared_k": "k",
                    },
                },
                {
                    "type": "c",
                    "target": "func_c2",
                    "params": {
                        "y": "c1",
                    }
                },
            ],
        }


def test_pipeline_planer_wandb_integration(planer_toy_registry):
    r = planer_toy_registry

    cfg = {
        "type": "a",
        "tune_mode": "pipeline",
        "pipeline": [
            {
                "type": "b",
            },
        ],
        "wandb": {
            "entity": "dance",
            "project": "dance-dev",
            "method": "bayes",
            "metric": {
                "name": "val/acc",
                "goal": "maximize",
            },
        },
    }
    p = PipelinePlaner(cfg, _registry=r)

    assert p.base_config == {
        "type": "a",
        "pipeline": [
            {
                "type": "b",
            },
        ],
    }
    assert p.wandb_config == {
        "entity": "dance",
        "project": "dance-dev",
        "method": "bayes",
        "metric": {
            "name": "val/acc",
            "goal": "maximize",
        },
    }

    assert dict(p.generate_config(pipeline=["func_b1"])) == {
        "type": "a",
        "pipeline": [
            {
                "type": "b",
                "target": "func_b1",
            },
        ],
    }

    assert p.search_space() == {
        "pipeline.0.b": {
            "values": ["func_b0", "func_b1", "func_b2"],
        },
    }

    assert p.wandb_sweep_config() == {
        "entity": "dance",
        "project": "dance-dev",
        "method": "bayes",
        "metric": {
            "name": "val/acc",
            "goal": "maximize",
        },
        "parameters": {
            "pipeline.0.b": {
                "values": ["func_b0", "func_b1", "func_b2"],
            },
        },
    }


def test_pipeline_subset():
    command_str, configs = generate_subsets("examples/tuning/cta_svm/pipeline_tuning_config.yaml", "pipeline",
                                            "examples/tuning/cta_svm/config_yamls/pipeline",
                                            "examples/tuning/cta_svm/main.py", "temp_data", [-1], False)
    assert repr(command_str) == repr("""#!/bin/bash
log_dir=temp_data
mkdir -p ${log_dir}
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_0_ --count=4 > temp_data/0.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_1_ --count=7 > temp_data/1.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_2_ --count=4 > temp_data/2.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_3_ --count=28 > temp_data/3.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_4_ --count=16 > temp_data/4.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_5_ --count=28 > temp_data/5.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_6_ --count=112 > temp_data/6.log 2>&1 &
""")
