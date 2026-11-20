"""Exact, persistent cache for deterministic preprocessing prefixes.

The cache stores each modality as an independent H5AD file and keeps the
``Data`` split indices in a small sidecar. Avoiding a monolithic H5MU file is
important here because these datasets have more than one million global
features across modalities. Keys include the input file metadata, concrete
action configuration, and the source files that implement those actions so
stale entries are not reused.

"""

import fcntl
import gc
import hashlib
import inspect
import json
import os
import pickle
import platform
import shutil
import tempfile
from pathlib import Path

import anndata
import mudata
import numpy
import scipy

from dance import logger
from dance.data import Data

CACHE_SCHEMA = "scmogcn-prefix-v2"


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_prefix(pipeline, depth):
    """Return the stable JSON representation of the first ``depth`` actions."""
    actions = [pipeline[index].to_dict() for index in range(min(depth, len(pipeline)))]
    return json.dumps(actions, sort_keys=True, separators=(",", ":"), default=str)


def _source_signature(dataset, pipeline, depth):
    source_paths = {Path(inspect.getsourcefile(dataset.__class__)).resolve()}
    for index in range(min(depth, len(pipeline))):
        target = pipeline[index]._get_target()
        source_path = inspect.getsourcefile(target)
        if source_path is not None:
            source_paths.add(Path(source_path).resolve())
    return [{"path": str(path), "sha256": _sha256_file(path)} for path in sorted(source_paths)]


def cache_identity(dataset, pipeline, depth, seed):
    input_files = []
    for raw_path in dataset.data_paths:
        path = Path(raw_path).resolve()
        stat = path.stat()
        input_files.append({
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })

    identity = {
        "schema": CACHE_SCHEMA,
        "subtask": dataset.subtask,
        "preprocess": dataset.preprocess,
        "normalize": dataset.normalize,
        "seed": seed,
        "depth": depth,
        "prefix": json.loads(canonical_prefix(pipeline, depth)),
        "input_files": input_files,
        "source_files": _source_signature(dataset, pipeline, depth),
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "anndata": anndata.__version__,
            "mudata": mudata.__version__,
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), identity


class PipelinePrefixCache:
    """Load or atomically build one concrete pipeline-prefix cache entry."""

    def __init__(self, root, dataset, pipeline, depth, seed):
        self.key, self.identity = cache_identity(dataset, pipeline, depth, seed)
        self.cache_dir = Path(root).resolve() / dataset.subtask / f"depth-{depth}"
        self.cache_path = self.cache_dir / f"{self.key}.cache"
        self.metadata_path = self.cache_path / "metadata.pkl"
        self.lock_path = self.cache_dir / f"{self.key}.lock"

    def _load(self):
        with self.metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("prefix cache metadata is not a dictionary")
        if metadata.get("schema") != CACHE_SCHEMA or metadata.get("key") != self.key:
            raise ValueError("prefix cache metadata does not match the requested key")
        modalities = {
            mod_name: anndata.read_h5ad(self.cache_path / f"{index:02d}-{mod_name}.h5ad")
            for index, mod_name in enumerate(metadata["mod_names"])
        }
        mdata = mudata.MuData(modalities)
        mdata.uns.update(metadata["global_uns"])
        data = Data(mdata)
        data._split_idx_dict = metadata["split_idx_dict"]
        return data

    def _move_invalid_cache(self):
        if not self.cache_path.exists():
            return
        suffix = 0
        while True:
            invalid_path = self.cache_path.with_name(f"{self.cache_path.name}.invalid-{os.getpid()}-{suffix}")
            if not invalid_path.exists():
                os.replace(self.cache_path, invalid_path)
                logger.warning(f"Moved invalid prefix cache to {invalid_path}")
                return
            suffix += 1

    def load_or_build(self, builder):
        """Return ``(data, hit)`` while serializing builders per exact key."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            if self.cache_path.exists() and self.metadata_path.exists():
                try:
                    data = self._load()
                    logger.info(f"PREFIX_CACHE_HIT key={self.key} path={self.cache_path}")
                    return data, True
                except Exception:
                    logger.exception(f"Unable to load prefix cache {self.cache_path}; rebuilding it")
                    self._move_invalid_cache()

            logger.info(f"PREFIX_CACHE_MISS key={self.key} path={self.cache_path}")
            data = builder()
            temp_cache_path = None
            try:
                temp_cache_path = Path(tempfile.mkdtemp(prefix=f".{self.key}.", suffix=".cache.tmp",
                                                        dir=self.cache_dir))
                mod_names = list(data.data.mod)
                for index, mod_name in enumerate(mod_names):
                    data.data.mod[mod_name].write_h5ad(temp_cache_path / f"{index:02d}-{mod_name}.h5ad")
                with (temp_cache_path / "metadata.pkl").open("wb") as handle:
                    pickle.dump(
                        {
                            "schema": CACHE_SCHEMA,
                            "key": self.key,
                            "identity": self.identity,
                            "mod_names": mod_names,
                            "global_uns": dict(data.data.uns),
                            "split_idx_dict": data._split_idx_dict,
                        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_cache_path, self.cache_path)
                cache_bytes = sum(path.stat().st_size for path in self.cache_path.iterdir())
                logger.info(f"PREFIX_CACHE_SAVED key={self.key} path={self.cache_path} bytes={cache_bytes}")
            except Exception:
                logger.exception("Failed to save prefix cache; continuing with the in-memory data")
                if temp_cache_path is not None:
                    shutil.rmtree(temp_cache_path, ignore_errors=True)
            finally:
                gc.collect()
            return data, False
