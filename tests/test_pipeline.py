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
