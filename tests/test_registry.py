from functools import partial

import pytest

from dance.registry import DotDict, Registry, register


def test_dotdict(subtests):
    with subtests.test("Empty"):
        dd = DotDict()

    with subtests.test("Single level"):
        dd = DotDict({"a": 1, "b": 2, "c": 3})
        assert dd["a"] == dd.a == 1
        assert dd["b"] == dd.b == 2
        assert dd["c"] == dd.c == 3

    with subtests.test("Multilevel"):
        dd = DotDict({"a": {"b": {"c": 1}}})
        assert dd.a.b.c == dd["a"]["b"]["c"] == 1
        assert dd.a.b.c == dd.a.b["c"] == dd["a"].b.c == dd.a["b"].c

    with subtests.test("Multilevel get"):
        dd = DotDict({"a": {"b": {"c": 1}}})
        assert dd.get("a.b.c") == dd.get("a.b")["c"] == dd.get("a.b").c == 1

        assert dd.get("a.b.d") is None
        with pytest.raises(KeyError):
            dd.get("a.b.d", missed_ok=False)

    with subtests.test("Multilevel get create default"):
        dd = DotDict({"a": {"b": {"c": 1}}})

        node = dd.get("x.y.z", create_on_miss=True)
        assert dict(dd.x.y.z) == dict(dd.x.y.z) == dict()
        dd.x.y.z["test"] = 2  # make sure reference is hooked up
        assert dict(node) == dict(dd.x.y.z) == dict(test=2)

        with pytest.raises(ValueError):
            dd.get("d.e", missed_ok=False, create_on_miss=True)

    with subtests.test("Multilevel set"):
        dd = DotDict({"a": {"b": {"c": 1}}})

        dd.set("a.b.d", 2)
        assert dd.get("a.b.d", missed_ok=False) == 2

        dd.set("a.b.d", 3, exist_ok=True)
        assert dd.get("a.b.d", missed_ok=False) == 3

        with pytest.raises(KeyError):
            dd.set("a.b.d", 3, exist_ok=False)

        with pytest.raises(KeyError):
            dd.set("a.b.c.d", 4)


def test_registry(subtests):
    with subtests.test("Empty"):
        r = Registry()

    with subtests.test("Leaf node"):
        r = Registry({"a": 1, "b": {"c": 2}})

        assert r.is_leaf_node("a")
        assert not r.is_leaf_node("b")
        assert r.is_leaf_node("b.c")

    with subtests.test("Children nodes"):
        r = Registry({"a": 1, "b": {"c": 2}})

        assert sorted(r.children(leaf_node=True, non_leaf_node=True)) == ["a", "b", "b.c"]
        assert sorted(r.children("b", leaf_node=True, non_leaf_node=True)) == ["b.c"]

        assert sorted(r.children(leaf_node=False, non_leaf_node=True)) == ["b"]
        assert sorted(r.children(leaf_node=True, non_leaf_node=False)) == ["a", "b.c"]

        with pytest.raises(KeyError):
            list(r.children("a"))

        assert list(r.children(leaf_node=True, non_leaf_node=False, return_val=True)) == [("a", 1), ("b.c", 2)]


def test_register(subtests):
    r1 = Registry()
    r2 = Registry()

    register("a", name="test", _registry=r1)(1)
    assert dict(r1) == {"a": {"test": 1}}
    assert dict(r2) == {}

    register("a.b", name="test", _registry=r1)(2)
    assert dict(r1) == {"a": {"test": 1, "b": {"test": 2}}}
    assert dict(r2) == {}

    register("a", "c", "d", name="test", _registry=r1)(3)
    assert dict(r1) == {"a": {"test": 1, "b": {"test": 2}, "c": {"d": {"test": 3}}}}
    assert dict(r2) == {}

    register_b = partial(register, "b")
    register_b("c", "d", name="test", _registry=r2)(4)
    assert dict(r1) == {"a": {"test": 1, "b": {"test": 2}, "c": {"d": {"test": 3}}}}
    assert dict(r2) == {"b": {"c": {"d": {"test": 4}}}}
