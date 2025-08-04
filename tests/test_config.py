import pytest
from omegaconf import DictConfig

from dance.config import Config


@pytest.fixture
def cfg_dict():
    return {"a": 1, "b": {"c": 2}, "d": {}}


def test_config_init(cfg_dict, subtests):
    with subtests.test("Empty"):
        cfg = Config()
        assert dict(cfg) == {}

    with subtests.test("From dict"):
        cfg = Config(cfg_dict)
        assert dict(cfg) == cfg_dict

    with subtests.test("From DictConfig"):
        cfg = Config(DictConfig(cfg_dict))
        assert dict(cfg) == cfg_dict


def test_config_dump(cfg_dict, tmp_path, subtests):
    cfg = Config(cfg_dict)

    path = tmp_path / "cfg.yaml"
    with subtests.test("Dump YAML", path=path):
        cfg.dump_yaml(path)
        ans = "a: 1\nb:\n  c: 2\nd: {}\n"
        with open(path) as f:
            assert f.read() == ans

    path = tmp_path / "cfg.json"
    with subtests.test("Dump JSON", path=path):
        cfg.dump_json(path)
        ans = '{\n    "a": 1,\n    "b": {\n        "c": 2\n    },\n    "d": {}\n}'
        with open(path) as f:
            assert f.read() == ans
