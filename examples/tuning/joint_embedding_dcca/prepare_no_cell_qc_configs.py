#!/usr/bin/env python
"""Prepare comparable DCCA reruns with threshold-based cell QC disabled."""

from __future__ import annotations

import argparse
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
CELL_QC_INDICES = (9, 10, 12, 13)
CELL_QC_TARGET = "FilterCellsPlaceHolder"
TARGET_ALIASES = {"ScaleFeature": "ColumnSumNormalize"}
PIPELINE_KEY_RE = re.compile(r"^pipeline\.(\d+)\.")


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


def canonicalize_run(run: dict) -> dict:
    canonical = {
        key: TARGET_ALIASES.get(value, value)
        for key, value in sorted(run.items(), key=lambda item: pipeline_index(item[0]))
    }
    for idx in CELL_QC_INDICES:
        key = f"pipeline.{idx}.filter.cell"
        if key not in canonical:
            raise KeyError(f"Missing required cell-QC choice {key!r}")
        canonical[key] = CELL_QC_TARGET
    return canonical


def deduplicate_runs(runs: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for raw_run in runs:
        run = canonicalize_run(raw_run)
        key = tuple(run.items())
        if key not in seen:
            seen.add(key)
            unique.append(run)
    return unique


def fix_cell_qc_search_space(config: dict) -> None:
    for idx in CELL_QC_INDICES:
        element = config["pipeline"][idx]
        if element.get("type") != "filter.cell" or "include" not in element:
            raise ValueError(f"Pipeline element {idx} is not a tunable cell filter: {element}")
        element["include"] = [CELL_QC_TARGET]
        defaults = element.get("default_params")
        if defaults is not None:
            defaults.pop("FilterCellsScanpyOrder", None)
            if not defaults:
                element.pop("default_params")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    paths = {task: base_dir / task / "pipeline_params_tuning_config.yaml" for task in TASKS}
    configs = {task: load_config(path) for task, path in paths.items()}

    source_lists = [configs[task].get("run_kwargs", []) for task in SOURCE_TASKS]
    if not source_lists[0] or source_lists[0] != source_lists[1]:
        raise ValueError("SKIN and openproblems must contain the same non-empty inherited BRAIN run list.")

    runs = deduplicate_runs(source_lists[0])
    total_runs = len(runs)
    if args.sample_size is not None:
        if not 0 < args.sample_size <= total_runs:
            raise ValueError(f"sample-size must be between 1 and {total_runs}, got {args.sample_size}")
        selected_indices = sorted(random.Random(args.seed).sample(range(total_runs), args.sample_size))
        runs = [runs[idx] for idx in selected_indices]

    for task, config in configs.items():
        fix_cell_qc_search_space(config)
        config["run_kwargs"] = runs
        config["run_kwargs_selection"] = {
            "source_count": total_runs,
            "selected_count": len(runs),
            "random_seed": args.seed if args.sample_size is not None else None,
            "cell_qc_indices": list(CELL_QC_INDICES),
            "cell_qc_target": CELL_QC_TARGET,
        }
        config["wandb"]["method"] = "grid"
        if not args.dry_run:
            with paths[task].open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False, width=120)

    action = "Would write" if args.dry_run else "Wrote"
    selection = f" sampled from {total_runs} with seed {args.seed}" if args.sample_size is not None else ""
    print(f"{action} {len(runs)} unique no-cell-QC pipelines{selection} to {len(paths)} task configs.")


if __name__ == "__main__":
    main()
