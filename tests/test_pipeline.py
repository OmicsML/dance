import pytest

from dance.pipeline import Action, Pipeline
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
