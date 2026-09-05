#!/usr/bin/env python
"""Prepare comparable DCCA reruns with threshold-based cell QC disabled."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TASKS = (
    "GSE140203_BRAIN_atac2gex",
    "GSE140203_SKIN_atac2gex",
    "openproblems_2022_multi_atac2gex",
)
SOURCE_TASKS = TASKS[1:]
CELL_QC_TARGET = "FilterCellsPlaceHolder"
TARGET_ALIASES = {"ScaleFeature": "ColumnSumNormalize"}
PIPELINE_KEY_RE = re.compile(r"^pipeline\.(\d+)\.")
EXPECTED_QC_MODS = ("mod1", "mod2", "meta1", "meta2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix DCCA cell-QC slots to placeholders and share one deduplicated run list across tasks.")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--sample-size", type=int, default=None, help="Select this many pipelines after deduplication.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed used with --sample-size.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pipeline_index(key: str) -> int:
    match = PIPELINE_KEY_RE.match(key)
    if match is None:
        raise ValueError(f"Unexpected run_kwargs key: {key!r}")
    return int(match.group(1))


def canonicalize_run(run: dict, cell_qc_indices: tuple[int, ...]) -> dict:
    canonical = {
        key: TARGET_ALIASES.get(value, value)
        for key, value in sorted(run.items(), key=lambda item: pipeline_index(item[0]))
    }
    for idx in cell_qc_indices:
        key = f"pipeline.{idx}.filter.cell"
        if key not in canonical:
            raise KeyError(f"Missing required cell-QC choice {key!r}")
        canonical[key] = CELL_QC_TARGET
    return canonical


def deduplicate_runs(runs: list[dict], cell_qc_indices: tuple[int, ...]) -> list[dict]:
    unique = []
    seen = set()
    for raw_run in runs:
        run = canonicalize_run(raw_run, cell_qc_indices)
        key = tuple(run.items())
        if key not in seen:
            seen.add(key)
            unique.append(run)
    return unique


def find_cell_qc_indices(config: dict) -> tuple[int, ...]:
    """Identify tunable threshold-based cell QC slots from their structure."""
    found = {}
    common_mod_indices = []
    for idx, element in enumerate(config.get("pipeline", [])):
        if element.get("type") != "filter.cell":
            continue
        if element.get("target") == "FilterCellsCommonMod":
            common_mod_indices.append(idx)
            continue
        include = element.get("include") or []
        mod = (element.get("params") or {}).get("mod")
        if "FilterCellsScanpyOrder" in include and CELL_QC_TARGET in include and mod in EXPECTED_QC_MODS:
            if mod in found:
                raise ValueError(f"Duplicate dynamic cell-QC slot for {mod!r}: {found[mod]} and {idx}")
            found[mod] = idx

    missing = [mod for mod in EXPECTED_QC_MODS if mod not in found]
    if missing:
        raise ValueError(f"Missing dynamic cell-QC slots for modalities {missing}; found={found}")
    if len(common_mod_indices) != 2:
        raise ValueError(f"Expected two FilterCellsCommonMod alignment steps, found {common_mod_indices}")
    return tuple(found[mod] for mod in EXPECTED_QC_MODS)


def fix_cell_qc_search_space(config: dict, cell_qc_indices: tuple[int, ...]) -> None:
    for idx in cell_qc_indices:
        element = config["pipeline"][idx]
        if element.get("type") != "filter.cell" or "include" not in element:
            raise ValueError(f"Pipeline element {idx} is not a tunable cell filter: {element}")
        element["include"] = [CELL_QC_TARGET]
        defaults = element.get("default_params")
        if defaults is not None:
            defaults.pop("FilterCellsScanpyOrder", None)
            if not defaults:
                element.pop("default_params")


def resolve_shared_source_runs(source_lists: list[list[dict]], cell_qc_indices: tuple[int, ...]) -> list[dict]:
    """Resolve the common inherited list, tolerating a proven extra prefix in one
    task."""
    canonical_lists = [[canonicalize_run(run, cell_qc_indices) for run in runs] for runs in source_lists]
    if not all(canonical_lists):
        raise ValueError("SKIN and openproblems must contain non-empty inherited BRAIN run lists.")
    if canonical_lists[0] == canonical_lists[1]:
        return canonical_lists[0]

    shorter, longer = sorted(canonical_lists, key=len)
    if len(longer) > len(shorter) and longer[-len(shorter):] == shorter:
        extra_count = len(longer) - len(shorter)
        print(
            f"Detected and excluded {extra_count} task-specific prefix runs; using the shared {len(shorter)}-run suffix."
        )
        return shorter
    raise ValueError("SKIN/openproblems run lists differ and neither is an exact canonicalized suffix of the other.")


def ordered_runs_sha256(runs: list[dict]) -> str:
    payload = json.dumps(runs, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonicalize_pipeline_search_space(config: dict) -> None:
    """Apply target aliases to pipeline candidates and target-specific defaults."""
    for element in config.get("pipeline", []):
        if "target" in element:
            element["target"] = TARGET_ALIASES.get(element["target"], element["target"])
        if "include" in element:
            element["include"] = [TARGET_ALIASES.get(target, target) for target in element["include"]]
        defaults = element.get("default_params")
        if defaults:
            element["default_params"] = {
                TARGET_ALIASES.get(target, target): params
                for target, params in defaults.items()
            }


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    paths = {task: base_dir / task / "pipeline_params_tuning_config.yaml" for task in TASKS}
    configs = {task: load_config(path) for task, path in paths.items()}

    indices_by_task = {task: find_cell_qc_indices(config) for task, config in configs.items()}
    if len(set(indices_by_task.values())) != 1:
        raise ValueError(f"Dynamic cell-QC indices differ across tasks: {indices_by_task}")
    cell_qc_indices = next(iter(indices_by_task.values()))
    print(f"Verified dynamic cell-QC indices {cell_qc_indices}; FilterCellsCommonMod alignment steps remain separate.")

    source_lists = [configs[task].get("run_kwargs", []) for task in SOURCE_TASKS]
    source_runs = resolve_shared_source_runs(source_lists, cell_qc_indices)
    source_count = len(source_runs)
    runs = deduplicate_runs(source_runs, cell_qc_indices)
    deduplicated_count = len(runs)
    if args.sample_size is not None:
        if not 0 < args.sample_size <= deduplicated_count:
            raise ValueError(f"sample-size must be between 1 and {deduplicated_count}, got {args.sample_size}")
        selected_indices = sorted(random.Random(args.seed).sample(range(deduplicated_count), args.sample_size))
        runs = [runs[idx] for idx in selected_indices]

    run_sha256 = ordered_runs_sha256(runs)

    for task, config in configs.items():
        canonicalize_pipeline_search_space(config)
        fix_cell_qc_search_space(config, cell_qc_indices)
        config["run_kwargs"] = runs
        config["run_kwargs_selection"] = {
            "source_count": source_count,
            "deduplicated_count": deduplicated_count,
            "selected_count": len(runs),
            "random_seed": args.seed if args.sample_size is not None else None,
            "ordered_run_kwargs_sha256": run_sha256,
            "cell_qc_indices": list(cell_qc_indices),
            "cell_qc_target": CELL_QC_TARGET,
        }
        config["wandb"]["method"] = "grid"
        if not args.dry_run:
            with paths[task].open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False, width=120)

    action = "Would write" if args.dry_run else "Wrote"
    selection = f" sampled from {deduplicated_count} with seed {args.seed}" if args.sample_size is not None else ""
    print(f"{action} {len(runs)} unique no-cell-QC pipelines{selection} to {len(paths)} task configs; "
          f"source={source_count}, sha256={run_sha256}.")


if __name__ == "__main__":
    main()
