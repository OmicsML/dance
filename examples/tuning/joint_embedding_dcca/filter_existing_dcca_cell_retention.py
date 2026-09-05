#!/usr/bin/env python
"""Annotate already-run DCCA pipeline sweeps with cell-retention stats.

This script is intentionally separate from ``main.py``: it reloads the dataset,
replays only preprocessing pipelines that already appear in the step-2 CSV, and
does not train the DCCA model.

"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba-cache-{os.environ.get('USER', 'user')}")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl-cache-{os.environ.get('USER', 'user')}")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def disable_numba_disk_cache() -> None:
    """Force third-party ``@numba.njit(cache=True)`` decorators to avoid disk cache."""
    try:
        import numba
    except Exception:
        return

    original_jit = numba.jit
    original_njit = numba.njit

    @wraps(original_jit)
    def jit_no_cache(*args, **kwargs):
        if kwargs.get("cache") is True:
            kwargs = dict(kwargs)
            kwargs["cache"] = False
        return original_jit(*args, **kwargs)

    @wraps(original_njit)
    def njit_no_cache(*args, **kwargs):
        if kwargs.get("cache") is True:
            kwargs = dict(kwargs)
            kwargs["cache"] = False
        return original_njit(*args, **kwargs)

    numba.jit = jit_no_cache
    numba.njit = njit_no_cache


disable_numba_disk_cache()

import pandas as pd
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dance.transforms  # noqa: E402,F401
from dance import logger  # noqa: E402
from dance.pipeline import PipelinePlaner, get_step3_yaml  # noqa: E402

DEFAULT_TASKS = ("GSE140203_BRAIN_atac2gex", "GSE140203_SKIN_atac2gex", "openproblems_2022_multi_atac2gex")
DEFAULT_GUARD_MODS = ("mod1", "mod2", "meta1", "meta2", "test_sol")
DEFAULT_DATA_ROOT = Path("/mnt/nfs/zyxing/data")
PIPELINE_PREFIXES = ("pipeline.", "run_kwargs_pipeline.")
TARGET_ALIASES = {
    "ScaleFeature": "ColumnSumNormalize",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay already-run DCCA preprocessing pipelines and filter sweeps by cell retention.")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR,
                        help="Directory containing DCCA task subdirectories.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Task directories to process.")
    parser.add_argument("--result-name", default="results/pipeline/best_test_acc.csv",
                        help="Step-2 result CSV path relative to each task directory.")
    parser.add_argument("--config-name", default="pipeline_params_tuning_config.yaml",
                        help="Pipeline tuning YAML path relative to each task directory.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="JointEmbeddingNIPSDataset root containing task directories.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed reset before each preprocessing replay (matches the tuning scripts).")
    parser.add_argument("--min-cell-retention", type=float, default=0.8,
                        help="Minimum min-modality cell-retention fraction for filtered outputs.")
    parser.add_argument("--min-cell-count", type=int, default=None,
                        help="Optional minimum min-modality kept cell count for filtered outputs.")
    parser.add_argument("--metric", default="ARI", help="Metric used for top-k and step3 YAML generation.")
    parser.add_argument("--ascending", action="store_true", help="Sort metric in ascending order.")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True,
                        help="Deduplicate rows by the full preprocessing pipeline before retention filtering.")
    parser.add_argument("--guard-mods", nargs="+", default=list(DEFAULT_GUARD_MODS),
                        help="Modalities used to compute preprocess.cell_retention.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only replay the first N unique pipelines, for smoke testing.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse existing retention detail CSV rows with no recorded error.")
    parser.add_argument(
        "--retry-errors",
        nargs="+",
        default=None,
        help="When resuming, retry only existing rows whose preprocess.error exactly matches one of these values. "
        "Existing rows with other errors are retained without replaying; unseen pipelines still run.",
    )
    parser.add_argument(
        "--retry-all-errors", action="store_true",
        help="When resuming, retry every existing preprocessing error while preserving successful rows.")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True,
                        help="Record failed preprocessing pipelines and continue.")
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True,
                        help="Back up existing ignored result/config files before writing outputs.")
    parser.add_argument("--backup-dir", type=Path, default=SCRIPT_DIR / "cell_retention_backups",
                        help="Root directory for timestamped backups.")
    parser.add_argument("--replace-result", action="store_true",
                        help="Replace the original result CSV with the filtered CSV after writing sidecars.")
    parser.add_argument("--write-step3-yamls", action="store_true",
                        help="Regenerate config_yamls/params from the annotated CSV with the retention guard.")
    parser.add_argument("--conf-load-path", type=Path, default=SCRIPT_DIR.parent / "step3_default_params.yaml",
                        help="Step3 parameter-search YAML passed to get_step3_yaml.")
    parser.add_argument(
        "--prefer-annotated-source", action=argparse.BooleanOptionalAction, default=True,
        help="If a full best_test_acc.with_cell_retention.csv exists, use it as the run-level source "
        "instead of an already replaced/filtered best_test_acc.csv.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read inputs and report planned work without replaying pipelines or writing outputs.")
    return parser.parse_args()


def task_dir(base_dir: Path, task: str) -> Path:
    return (base_dir / task).resolve()


def ensure_script_cwd() -> None:
    os.chdir(SCRIPT_DIR)


def pipeline_col_index(col: str) -> int:
    for prefix in PIPELINE_PREFIXES:
        if col.startswith(prefix):
            return int(col[len(prefix):].split(".", 1)[0])
    raise ValueError(f"Not a pipeline column: {col}")


def pipeline_columns(result: pd.DataFrame) -> list[str]:
    cols = [col for col in result.columns if col.startswith(PIPELINE_PREFIXES)]
    return sorted(cols, key=lambda col: (pipeline_col_index(col), col))


def canonical_pipeline_target(value) -> str:
    value = str(value)
    return TARGET_ALIASES.get(value, value)


def canonicalize_aliases(value):
    if isinstance(value, str):
        return canonical_pipeline_target(value)
    if isinstance(value, list):
        return [canonicalize_aliases(item) for item in value]
    if isinstance(value, dict):
        canonical = {}
        for key, item in value.items():
            canonical_key = canonical_pipeline_target(key) if isinstance(key, str) else key
            canonical_value = canonicalize_aliases(item)
            if canonical_key not in canonical:
                canonical[canonical_key] = canonical_value
        return canonical
    return value


def load_pipeline_config(config_path: Path):
    conf = OmegaConf.load(config_path)
    return OmegaConf.create(canonicalize_aliases(OmegaConf.to_container(conf, resolve=False)))


def load_pipeline_planer(config_path: Path) -> PipelinePlaner:
    return PipelinePlaner(load_pipeline_config(config_path))


def canonicalize_pipeline_columns(result: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    canonical = result.copy()
    for col in cols:
        canonical[col] = canonical[col].map(canonical_pipeline_target)
    return canonical


def normalize_pipeline_column(col: str) -> str:
    if col.startswith("run_kwargs_pipeline."):
        return col.replace("run_kwargs_pipeline.", "pipeline.", 1)
    return col


def pipeline_key(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def unique_pipeline_table(result: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    unique = canonicalize_pipeline_columns(result.loc[:, list(cols)],
                                           cols).dropna().drop_duplicates().reset_index(drop=True)
    unique.insert(0, "preprocess.pipeline_key", [pipeline_key(row) for row in unique.astype(str).to_numpy()])
    return unique


def strip_preprocess_columns(result: pd.DataFrame) -> pd.DataFrame:
    stale_cols = [col for col in result.columns if col.startswith("preprocess.")]
    return result.drop(columns=stale_cols) if stale_cols else result


def read_run_level_result(result_path: Path, prefer_annotated_source: bool) -> tuple[pd.DataFrame, Path]:
    result = pd.read_csv(result_path)
    source_path = result_path
    if prefer_annotated_source:
        annotated_path = result_path.parent / "best_test_acc.with_cell_retention.csv"
        if annotated_path.exists():
            annotated = pd.read_csv(annotated_path)
            if len(annotated) > len(result) and pipeline_columns(annotated):
                logger.info(f"Using {annotated_path} as the run-level source because it has {len(annotated)} rows; "
                            f"{result_path} has {len(result)} rows.")
                result = annotated
                source_path = annotated_path
    return strip_preprocess_columns(result), source_path


def get_mods(data) -> dict[str, object]:
    if hasattr(data, "mod"):
        return data.mod
    return data.data.mod


def mod_shape_table(data) -> dict[str, dict[str, int]]:
    shapes = {}
    for name, adata in get_mods(data).items():
        shapes[name] = {"n_cells": int(adata.n_obs), "n_features": int(adata.n_vars)}
    return shapes


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def retention_record(raw_shapes: dict[str, dict[str, int]], kept_shapes: dict[str, dict[str, int]],
                     guard_mods: Iterable[str], prefix: str = "preprocess") -> dict[str, object]:
    record: dict[str, object] = {}
    guard_retentions = []
    guard_counts = []
    for mod in sorted(raw_shapes):
        raw_cells = raw_shapes[mod]["n_cells"]
        kept_cells = kept_shapes.get(mod, {}).get("n_cells", 0)
        raw_features = raw_shapes[mod]["n_features"]
        kept_features = kept_shapes.get(mod, {}).get("n_features", 0)
        record[f"{prefix}.{mod}.raw_n_cells"] = raw_cells
        record[f"{prefix}.{mod}.n_cells"] = kept_cells
        record[f"{prefix}.{mod}.removed_n_cells"] = raw_cells - kept_cells
        record[f"{prefix}.{mod}.cell_retention"] = safe_fraction(kept_cells, raw_cells)
        record[f"{prefix}.{mod}.raw_n_features"] = raw_features
        record[f"{prefix}.{mod}.n_features"] = kept_features
        record[f"{prefix}.{mod}.feature_retention"] = safe_fraction(kept_features, raw_features)
        if mod in guard_mods:
            guard_retentions.append(record[f"{prefix}.{mod}.cell_retention"])
            guard_counts.append(kept_cells)

    record[f"{prefix}.min_cell_retention"] = min(guard_retentions) if guard_retentions else None
    record[f"{prefix}.cell_retention"] = record[f"{prefix}.min_cell_retention"]
    record[f"{prefix}.min_n_cells"] = min(guard_counts) if guard_counts else None
    record[f"{prefix}.guard_mods"] = ",".join(guard_mods)
    return record


def load_raw_data_once(task: str, data_root: Path):
    from dance.datasets.multimodality import JointEmbeddingNIPSDataset

    dataset = JointEmbeddingNIPSDataset(task, root=str(data_root))
    raw_data = dataset._load_raw_data()
    return dataset, raw_data


def clone_raw_data(raw_data):
    return [adata.copy() for adata in raw_data]


def run_preprocessing_once(dataset, raw_data, planer: PipelinePlaner, pipeline_spec: dict[str, str],
                           guard_mods: Sequence[str], seed: int) -> dict[str, object]:
    # The original tuning entry points reset this seed before each pipeline
    # evaluation.  This matters for ScTransform's gene sampling.
    from dance.utils import set_seed

    set_seed(seed)
    data = dataset._raw_to_dance(clone_raw_data(raw_data))
    raw_shapes = mod_shape_table(data)

    preprocessing_pipeline = planer.generate(pipeline_params=pipeline_spec)
    last_cell_filter_idx = max(idx for idx, elem in enumerate(preprocessing_pipeline.config.pipeline)
                               if elem.get("type") == "filter.cell")
    for idx in range(last_cell_filter_idx + 1):
        preprocessing_pipeline[idx](data)
        kept_shapes = mod_shape_table(data)
        # A pipeline that removes every cell from a guarded modality has a
        # well-defined retention of zero. Do not invoke later transforms that
        # expect a non-empty AnnData object merely to obtain that result.
        if any(kept_shapes.get(mod, {}).get("n_cells", 0) == 0 for mod in guard_mods):
            return retention_record(raw_shapes, kept_shapes, guard_mods)

    kept_shapes = mod_shape_table(data)
    return retention_record(raw_shapes, kept_shapes, guard_mods)


def backup_path(src: Path, task_backup_dir: Path, task_root: Path) -> Path:
    try:
        rel = src.resolve().relative_to(task_root.resolve())
    except ValueError:
        rel = Path(src.name)
    return task_backup_dir / rel


def copy_for_backup(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def backup_task_files(task_root: Path, backup_root: Path, extra_paths: Sequence[Path]) -> None:
    paths = [
        task_root / "pipeline_params_tuning_config.yaml",
        task_root / "config_yamls" / "params",
        task_root / "results" / "params",
        task_root / "results" / "pipeline" / "best_test_acc.csv",
    ]
    paths.extend(extra_paths)
    for src in paths:
        copy_for_backup(src, backup_path(src, backup_root, task_root))


def sort_for_metric(df: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    if metric not in df:
        return df
    ranked = df.copy()
    ranked[f"__{metric}_numeric"] = pd.to_numeric(ranked[metric], errors="coerce")
    return ranked.sort_values([f"__{metric}_numeric"], ascending=ascending,
                              na_position="last").drop(columns=[f"__{metric}_numeric"])


def dedupe_annotated_result(annotated: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    grouped_rows = []
    metric_cols = [col for col in ["ARI", "dance_nmi", metric] if col in annotated]
    metric_cols = list(dict.fromkeys(metric_cols))
    for _, group in annotated.groupby("preprocess.pipeline_key", sort=False, dropna=False):
        chosen = sort_for_metric(group, metric, ascending).iloc[0].copy()
        chosen["preprocess.duplicate_run_count"] = len(group)
        if "id" in group:
            chosen["preprocess.duplicate_run_ids"] = ";".join(group["id"].astype(str))
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            chosen[f"preprocess.{col}.mean_over_duplicate_runs"] = values.mean()
            chosen[f"preprocess.{col}.best_over_duplicate_runs"] = values.min() if ascending else values.max()
            chosen[f"preprocess.{col}.valid_duplicate_runs"] = int(values.notna().sum())
        grouped_rows.append(chosen)
    return pd.DataFrame(grouped_rows).reset_index(drop=True)


def write_outputs(task: str, task_root: Path, result: pd.DataFrame, detail: pd.DataFrame, cols: Sequence[str],
                  args: argparse.Namespace) -> dict[str, Path]:
    result_path = task_root / args.result_name
    pipeline_dir = result_path.parent
    retention_path = pipeline_dir / "preprocess_cell_retention_by_pipeline.csv"
    annotated_path = pipeline_dir / "best_test_acc.with_cell_retention.csv"
    deduped_path = pipeline_dir / "best_test_acc.deduped.with_cell_retention.csv"
    threshold_label = str(args.min_cell_retention).replace(".", "p")
    filtered_path = pipeline_dir / f"best_test_acc.deduped.cell_retention_ge_{threshold_label}.csv"
    topk_path = pipeline_dir / f"best_test_acc.deduped.cell_retention_ge_{threshold_label}.top_k.csv"
    run_level_filtered_path = pipeline_dir / f"best_test_acc.run_level.cell_retention_ge_{threshold_label}.csv"

    detail.to_csv(retention_path, index=False)

    keyed = result.copy()
    if "preprocess.pipeline_key" in keyed:
        keyed = keyed.drop(columns=["preprocess.pipeline_key"])
    keyed.insert(0, "preprocess.pipeline_key",
                 [pipeline_key(row) for row in keyed.loc[:, list(cols)].astype(str).to_numpy()])
    stat_cols = [col for col in detail.columns if col not in cols]
    annotated = keyed.merge(detail.loc[:, stat_cols], on="preprocess.pipeline_key", how="left")
    if "preprocess.cell_retention" not in annotated:
        annotated["preprocess.cell_retention"] = pd.NA
    if "preprocess.min_n_cells" not in annotated:
        annotated["preprocess.min_n_cells"] = pd.NA
    annotated.to_csv(annotated_path, index=False)

    run_level_mask = pd.to_numeric(annotated["preprocess.cell_retention"], errors="coerce") >= args.min_cell_retention
    if args.min_cell_count is not None:
        run_level_mask &= pd.to_numeric(annotated["preprocess.min_n_cells"], errors="coerce") >= args.min_cell_count
    annotated.loc[run_level_mask].copy().to_csv(run_level_filtered_path, index=False)

    selection_source = dedupe_annotated_result(annotated, args.metric, args.ascending) if args.dedupe else annotated
    selection_source.to_csv(deduped_path, index=False)

    mask = pd.to_numeric(selection_source["preprocess.cell_retention"], errors="coerce") >= args.min_cell_retention
    if args.min_cell_count is not None:
        mask &= pd.to_numeric(selection_source["preprocess.min_n_cells"], errors="coerce") >= args.min_cell_count
    filtered = selection_source.loc[mask].copy()
    filtered.to_csv(filtered_path, index=False)

    conf = load_pipeline_config(task_root / args.config_name)
    top_k = int(conf.get("pipeline_tuning_top_k", 3))
    if args.metric in filtered:
        topk = filtered.sort_values(args.metric, ascending=args.ascending).head(top_k)
    else:
        topk = filtered.head(top_k)
    topk.to_csv(topk_path, index=False)

    if args.replace_result:
        shutil.copy2(filtered_path, result_path)

    if args.write_step3_yamls:
        get_step3_yaml(
            result_load_path=str(deduped_path),
            step2_pipeline_planer=load_pipeline_planer(task_root / args.config_name),
            conf_load_path=str(args.conf_load_path),
            root_path=str(task_root),
            required_funs=["AlignMod", "FilterCellsCommonMod", "FilterCellsCommonMod", "SetConfig"],
            required_indexes=[2, 11, 14, sys.maxsize],
            metric=args.metric,
            ascending=args.ascending,
            min_cell_retention=args.min_cell_retention,
            min_cell_count=args.min_cell_count,
        )

    logger.info(f"{task}: rows={len(result)}, deduped_rows={len(selection_source)}, unique_preprocess={len(detail)}, "
                f"retention_pass_deduped_rows={len(filtered)}, top_k_rows={len(topk)}")
    return {
        "retention": retention_path,
        "annotated": annotated_path,
        "deduped": deduped_path,
        "filtered": filtered_path,
        "run_level_filtered": run_level_filtered_path,
        "top_k": topk_path,
    }


def process_task(task: str, args: argparse.Namespace, timestamp_backup_root: Path | None) -> None:
    root = task_dir(args.base_dir, task)
    result_path = root / args.result_name
    config_path = root / args.config_name
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    result, source_path = read_run_level_result(result_path, args.prefer_annotated_source)
    cols = pipeline_columns(result)
    if not cols:
        raise ValueError(f"No pipeline columns found in {result_path}")
    result = canonicalize_pipeline_columns(result, cols)
    unique = unique_pipeline_table(result, cols)
    if args.limit is not None:
        unique = unique.head(args.limit)

    pipeline_dir = result_path.parent
    retention_path = pipeline_dir / "preprocess_cell_retention_by_pipeline.csv"
    output_paths = [
        retention_path,
        pipeline_dir / "best_test_acc.with_cell_retention.csv",
        pipeline_dir / "best_test_acc.deduped.with_cell_retention.csv",
        pipeline_dir / f"best_test_acc.deduped.cell_retention_ge_{str(args.min_cell_retention).replace('.', 'p')}.csv",
        pipeline_dir /
        f"best_test_acc.deduped.cell_retention_ge_{str(args.min_cell_retention).replace('.', 'p')}.top_k.csv",
        pipeline_dir /
        f"best_test_acc.run_level.cell_retention_ge_{str(args.min_cell_retention).replace('.', 'p')}.csv",
    ]

    if args.backup and timestamp_backup_root is not None:
        backup_task_files(root, timestamp_backup_root / task, output_paths)

    logger.info(f"{task}: result source={source_path}, result rows={len(result)}, "
                f"unique already-run pipelines={len(unique)}")
    if args.dry_run:
        logger.info(f"{task}: dry-run, skip preprocessing replay and output writes.")
        return

    planer = load_pipeline_planer(config_path)
    logger.info(f"{task}: loading raw data once from {args.data_root}")
    dataset, raw_data = load_raw_data_once(task, args.data_root)

    existing = pd.DataFrame()
    done_keys = set()
    if args.resume and retention_path.exists():
        existing = pd.read_csv(retention_path)
        if "preprocess.pipeline_key" in existing and "preprocess.error" in existing:
            errors = existing["preprocess.error"].fillna("").astype(str)
            if args.retry_all_errors:
                done = existing[errors == ""]
                logger.info(f"{task}: retrying all {int((errors != '').sum())} existing preprocessing errors")
            elif args.retry_errors:
                retry_errors = set(args.retry_errors)
                done = existing[(errors == "") | ~errors.isin(retry_errors)]
                logger.info(f"{task}: retrying existing errors only when they match {sorted(retry_errors)!r}")
            else:
                done = existing[errors == ""]
            done_keys = set(done["preprocess.pipeline_key"].astype(str))

    records: list[dict[str, object]] = []
    if not existing.empty:
        records.extend(existing.to_dict("records"))

    for idx, row in unique.iterrows():
        key = str(row["preprocess.pipeline_key"])
        if key in done_keys:
            continue
        pipeline_spec = {normalize_pipeline_column(col): str(row[col]) for col in cols}
        record: dict[str, object] = row.to_dict()
        logger.info(f"{task}: replay {idx + 1}/{len(unique)} pipeline_key={key}")
        try:
            record.update(run_preprocessing_once(dataset, raw_data, planer, pipeline_spec, args.guard_mods, args.seed))
            record["preprocess.error"] = ""
        except Exception as exc:
            record["preprocess.error"] = repr(exc)
            record["preprocess.traceback"] = traceback.format_exc()
            if not args.continue_on_error:
                raise
        records.append(record)
        pd.DataFrame(records).drop_duplicates("preprocess.pipeline_key",
                                              keep="last").to_csv(retention_path, index=False)
        gc.collect()

    detail = pd.DataFrame(records).drop_duplicates("preprocess.pipeline_key", keep="last")
    write_outputs(task, root, result, detail, cols, args)


def main() -> None:
    args = parse_args()
    ensure_script_cwd()
    timestamp_backup_root = None
    if args.backup and not args.dry_run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_backup_root = (args.backup_dir / stamp).resolve()
        timestamp_backup_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backups will be written under {timestamp_backup_root}")

    for task in args.tasks:
        process_task(task, args, timestamp_backup_root)


if __name__ == "__main__":
    main()
