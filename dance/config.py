import json
import os.path as osp
from typing import get_args

from omegaconf import DictConfig, OmegaConf

from dance.typing import Any, ConfigLike, Dict, FileExistHandle, Literal, Optional, PathLike
from dance.utils import file_check

CONFIG_FTYPE = Literal["json", "yaml"]


class Config(DictConfig):

    def __init__(self, content: Optional[ConfigLike] = None, **kwargs):
        super().__init__(content, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return OmegaConf.to_container(self, resolve=True)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self, resolve=True)

    def _dump_file(self, ftype: CONFIG_FTYPE, path: PathLike, exist_handle: FileExistHandle):
        file_check(path, exist_handle=exist_handle)
        with open(path, "w") as f:
            if ftype == "json":
                json.dump(self.to_dict(), f, indent=4)
            elif ftype == "yaml":
                f.write(self.to_yaml())
            else:
                raise ValueError(f"Unknwon dumping file type: {ftype}, supported options: {get_args(CONFIG_FTYPE)}")

    def dump_json(self, path: PathLike, exist_handle: FileExistHandle = "warn"):
        """Dump config file as JSON."""
        self._dump_file("json", path, exist_handle)

    def dump_yaml(self, path: PathLike, exist_handle: FileExistHandle = "warn"):
        """Dump config file as YAML."""
        self._dump_file("yaml", path, exist_handle)

    @classmethod
    def from_file(cls, path: PathLike, **kwargs):
        if not osp.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        cfg = OmegaConf.load(path)
        return cls(cfg, **kwargs)
