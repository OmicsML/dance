#!/usr/bin/env python
"""Summarize cell-filtering impact for clustering tuning YAMLs on one server.

Each tuning YAML can choose among gene-filter and cell-filter implementations during
search, so this reports all supported combinations from the include list.

"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

METHOD_DIR = {
    "Graphsc": "cluster_graphsc",
    "SCDCC": "cluster_scdcc",
    "SCDEEPCLUSTER": "cluster_scdeepcluster",
    "SCDSC": "cluster_scdsc",
    "SCTAG": "cluster_sctag",
}

DATASET_DIR = {
    "10X PBMC": "10X_PBMC",
    "mouse ES cell": "mouse_ES_cell",
    "human pbmc cell": "human_pbmc2_cell",
}


def dataset_dir(name: str) -> str:
    return DATASET_DIR.get(name, name.replace(" ", "_"))


def server_112_configs(tuning_dir: Path, server: str) -> list[tuple[str, str, Path]]:
    csv_path = tuning_dir / "results - cluster.csv"
    rows = list(csv.reader(csv_path.open(newline="")))
    header = rows[0]
    server_re = re.compile(r"^(?:\d+|papermachine)$")
    url_re = re.compile(r"https?://")

    current_method = None
    configs = []
    for row in rows[1:]:
        row = row + [""] * (len(header) - len(row))
        if row[0].strip():
            current_method = row[0].strip()

        values = [cell.strip() for cell in row[2:]]
        is_server_row = (any(server_re.fullmatch(value) for value in values)
                         and not any(url_re.search(value) for value in values))
        if not is_server_row:
            continue

        for col_idx, value in enumerate(values, start=2):
            if value != server:
                continue
            method_path = METHOD_DIR[current_method]
            dataset_path = dataset_dir(header[col_idx])
            config = tuning_dir / method_path / dataset_path / "pipeline_params_tuning_config.yaml"
            configs.append((current_method, dataset_path, config))
    return configs


def load_x(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        return np.asarray(handle["X"])


def clustering_urls(metadata_path: Path) -> dict[str, str]:
    with metadata_path.open(newline="") as handle:
        return {row[0]: row[1] for row in csv.reader(handle) if row}


def get_count(count_or_ratio, total: int):
    if count_or_ratio is None:
        return None
    if isinstance(count_or_ratio, float):
        if count_or_ratio > 1:
            raise ValueError(f"Ratio-like value is > 1: {count_or_ratio}")
        return int(count_or_ratio * total)
    return int(count_or_ratio)


def _counts_threshold(values: np.ndarray, threshold):
    if isinstance(threshold, float) and 0 < threshold < 1:
        return np.percentile(values, threshold * 100)
    return threshold


def filter_cell_mask(x: np.ndarray, name: str, value) -> np.ndarray:
    if value is None:
        return np.ones(x.shape[0], dtype=bool)
    _, n_genes = x.shape
    if name in {"min_counts", "max_counts"}:
        counts = np.asarray(x.sum(axis=1)).ravel()
        threshold = _counts_threshold(counts, value)
        return counts >= threshold if name == "min_counts" else counts <= threshold
    if name in {"min_genes", "max_genes"}:
        expressed = np.asarray((x > 0).sum(axis=1)).ravel()
        threshold = get_count(value, n_genes)
        return expressed >= threshold if name == "min_genes" else expressed <= threshold
    raise KeyError(name)


def filter_gene_mask(x: np.ndarray, name: str, value) -> np.ndarray:
    if value is None:
        return np.ones(x.shape[1], dtype=bool)
    n_cells, _ = x.shape
    if name in {"min_counts", "max_counts"}:
        counts = np.asarray(x.sum(axis=0)).ravel()
        threshold = _counts_threshold(counts, value)
        return counts >= threshold if name == "min_counts" else counts <= threshold
    if name in {"min_cells", "max_cells"}:
        expressed = np.asarray((x > 0).sum(axis=0)).ravel()
        threshold = get_count(value, n_cells)
        return expressed >= threshold if name == "min_cells" else expressed <= threshold
    raise KeyError(name)


def percentile_gene_mask(x: np.ndarray, params: dict) -> np.ndarray:
    mode = params.get("mode", "sum")
    if mode == "sum":
        summary = np.asarray(x.sum(axis=0)).ravel()
    elif mode == "var":
        summary = np.asarray(x.var(axis=0)).ravel()
    elif mode == "cv":
        mean = np.asarray(x.mean(axis=0)).ravel()
        std = np.asarray(x.std(axis=0)).ravel()
        summary = np.nan_to_num(std / mean, posinf=0, neginf=0)
    elif mode == "rv":
        mean = np.asarray(x.mean(axis=0)).ravel()
        var = np.asarray(x.var(axis=0)).ravel()
        summary = np.nan_to_num(var / mean, posinf=0, neginf=0)
    else:
        raise ValueError(f"Unsupported FilterGenesPercentile mode: {mode}")
    lo = np.percentile(summary, params.get("min_val", 1))
    hi = np.percentile(summary, params.get("max_val", 99))
    return np.logical_and(summary >= lo, summary <= hi)


def expand_gene_options(step: dict) -> list[tuple[str, dict | None]]:
    target = step.get("target")
    params = step.get("params") or {}
    defaults = step.get("default_params", {})
    names = [target] if target else step.get("include", [])
    options = []
    for name in names:
        if name == "FilterGenesPlaceHolder":
            options.append((name, None))
        elif name == "FilterGenesScanpyOrder":
            options.append((name, defaults.get(name, params)))
        elif name == "FilterGenesPercentile":
            options.append((name, defaults.get(name, params) or {"min_val": 1, "max_val": 99, "mode": "sum"}))
    return options or [(target or "NONE", params or None)]


def expand_cell_options(step: dict) -> list[tuple[str, dict | None]]:
    target = step.get("target")
    params = step.get("params") or {}
    defaults = step.get("default_params", {})
    names = [target] if target else step.get("include", [])
    options = []
    for name in names:
        if name == "FilterCellsPlaceHolder":
            options.append((name, None))
        elif name == "FilterCellsScanpyOrder":
            options.append((name, defaults.get(name, params)))
    return options or [(target or "NONE", params or None)]


def pipeline_before_and_at_cell_filter(config_path: Path):
    config = yaml.safe_load(config_path.read_text())
    gene_options = [("FilterGenesPlaceHolder", None)]
    for step in config.get("pipeline", []):
        if step.get("type") == "filter.gene":
            gene_options = expand_gene_options(step)
            continue
        if step.get("type") == "filter.cell":
            return gene_options, expand_cell_options(step)
    return gene_options, [("NONE", None)]


def download_h5(tuning_dir: Path, method_dir: str, dataset: str) -> Path | None:
    url = clustering_urls(tuning_dir.parent.parent / "dance" / "metadata" / "clustering.csv").get(dataset)
    if not url:
        return None
    # Clustering tuning entry points use ``--data_dir ../temp_data`` by
    # default, so downloaded data must go to the same shared directory.
    target = tuning_dir / "temp_data" / f"{dataset}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset} -> {target}", file=__import__("sys").stderr)
    urllib.request.urlretrieve(url, target)
    return target


def find_h5(tuning_dir: Path, method_dir: str, dataset: str, download: bool) -> Path | None:
    # Keep this order aligned with the clustering ``main.py`` defaults.  A
    # method-local data file can be stale or represent a different dataset
    # version even when it has the same filename.
    candidates = [
        tuning_dir / "temp_data" / f"{dataset}.h5",
        tuning_dir / method_dir / "temp_data" / f"{dataset}.h5",
        tuning_dir / method_dir / "data" / f"{dataset}.h5",
        tuning_dir.parent / "single_modality" / "clustering" / "data" / f"{dataset}.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in tuning_dir.glob(f"cluster_*/data/{dataset}.h5"):
        if path.exists():
            return path
    if download:
        return download_h5(tuning_dir, method_dir, dataset)
    return None


def summarize_one(tuning_dir: Path, method: str, dataset: str, config_path: Path,
                  download: bool) -> list[dict[str, str]]:
    method_dir = METHOD_DIR[method]
    h5_path = find_h5(tuning_dir, method_dir, dataset, download)
    gene_options, cell_options = pipeline_before_and_at_cell_filter(config_path)
    row = {
        "method": method,
        "dataset": dataset,
        "gene_filter": "",
        "cell_filter": "",
        "raw_cells": "NA",
        "cells_before_filter_cell": "NA",
        "kept_cells": "NA",
        "removed_cells": "NA",
        "removed_pct": "NA",
        "pre_gene_detail": "",
        "cell_step_detail": "",
        "h5": str(h5_path) if h5_path else "MISSING",
        "yaml": str(config_path),
    }
    if h5_path is None:
        row["scenario"] = "MISSING_DATA"
        return [row]

    raw_x = load_x(h5_path)
    raw_cells = raw_x.shape[0]
    raw_genes = raw_x.shape[1]

    rows = []
    for gene_target, gene_params in gene_options:
        x = raw_x
        gene_details = []
        if gene_params is None:
            gene_details.append(f"{gene_target}:skip")
        elif gene_target == "FilterGenesScanpyOrder":
            order = gene_params.get("order", ["min_counts", "min_cells", "max_counts", "max_cells"])
            for name in order:
                before = x.shape[1]
                mask = filter_gene_mask(x, name, gene_params.get(name))
                x = x[:, mask]
                after = x.shape[1]
                gene_details.append(f"{name}={gene_params.get(name)}:{before}->{after}(-{before - after})")
        elif gene_target == "FilterGenesPercentile":
            before = x.shape[1]
            mask = percentile_gene_mask(x, gene_params)
            x = x[:, mask]
            after = x.shape[1]
            gene_details.append(f"min_val={gene_params.get('min_val', 1)},max_val={gene_params.get('max_val', 99)},"
                                f"mode={gene_params.get('mode', 'sum')}:{before}->{after}(-{before - after})")
        else:
            gene_details.append(f"{gene_target}:unsupported")

        for cell_target, cell_params in cell_options:
            combo_row = row.copy()
            combo_row["gene_filter"] = gene_target
            combo_row["cell_filter"] = cell_target
            if cell_params is None:
                combo_row.update({
                    "scenario": f"{gene_target}+{cell_target}",
                    "raw_cells": str(raw_cells),
                    "raw_genes": str(raw_genes),
                    "cells_before_filter_cell": str(x.shape[0]),
                    "genes_before_filter_cell": str(x.shape[1]),
                    "kept_cells": str(x.shape[0]),
                    "removed_cells": str(raw_cells - x.shape[0]),
                    "removed_pct": f"{(raw_cells - x.shape[0]) / raw_cells:.2%}" if raw_cells else "NA",
                    "pre_gene_detail": f"raw_genes={raw_genes}; " + "; ".join(gene_details),
                    "cell_step_detail": f"{cell_target}:skip",
                })
            else:
                combo_row = summarize_cell_filter(
                    combo_row,
                    raw_cells=raw_cells,
                    raw_genes=raw_genes,
                    x=x.copy(),
                    params=cell_params,
                    pre_gene_detail=f"raw_genes={raw_genes}; " + "; ".join(gene_details),
                    scenario=f"{gene_target}+{cell_target}",
                )
            rows.append(combo_row)
    return rows


def annotate_summary(
    tuning_dir: Path,
    method: str,
    dataset: str,
    summary_file: Path,
    output: Path,
    download: bool,
) -> None:
    config_path = tuning_dir / METHOD_DIR[method] / dataset / "pipeline_params_tuning_config.yaml"
    combo_rows = summarize_one(tuning_dir, method, dataset, config_path, download=download)
    by_combo = {(row["gene_filter"], row["cell_filter"]): row for row in combo_rows}

    summary = pd.read_csv(summary_file)
    gene_cols = sorted([col for col in summary.columns if col.startswith("pipeline.") and col.endswith(".filter.gene")],
                       key=lambda col: float(col.split(".")[1]))
    cell_cols = sorted([col for col in summary.columns if col.startswith("pipeline.") and col.endswith(".filter.cell")],
                       key=lambda col: float(col.split(".")[1]))
    if not gene_cols or not cell_cols:
        raise ValueError(f"Could not find pipeline filter columns in {summary_file}")

    annotated_rows = []
    for _, row in summary.iterrows():
        gene_filter = row[gene_cols[0]]
        cell_filter = row[cell_cols[0]]
        stats = by_combo.get((gene_filter, cell_filter))
        annotated = row.to_dict()
        if stats is not None:
            annotated.update({
                "preprocess.raw_n_cells": int(stats["raw_cells"]),
                "preprocess.n_cells": int(stats["kept_cells"]),
                "preprocess.removed_n_cells": int(stats["removed_cells"]),
                "preprocess.cell_retention": int(stats["kept_cells"]) / int(stats["raw_cells"]),
                "preprocess.raw_n_features": int(stats["raw_genes"]),
                "preprocess.n_features_before_cell_filter": int(stats["genes_before_filter_cell"]),
            })
        annotated_rows.append(annotated)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(annotated_rows).to_csv(output, index=False)


def summarize_cell_filter(
    row: dict[str, str],
    raw_cells: int,
    raw_genes: int,
    x: np.ndarray,
    params: dict,
    pre_gene_detail: str,
    scenario: str,
) -> dict[str, str]:
    cells_before_filter_cell = x.shape[0]
    genes_before_filter_cell = x.shape[1]
    order = params.get("order", ["min_counts", "min_genes", "max_counts", "max_genes"])
    cell_details = []
    for name in order:
        before = x.shape[0]
        mask = filter_cell_mask(x, name, params.get(name))
        x = x[mask]
        after = x.shape[0]
        cell_details.append(f"{name}={params.get(name)}:{before}->{after}(-{before - after})")

    removed = raw_cells - x.shape[0]
    row.update({
        "scenario": scenario,
        "raw_cells": str(raw_cells),
        "raw_genes": str(raw_genes),
        "cells_before_filter_cell": str(cells_before_filter_cell),
        "genes_before_filter_cell": str(genes_before_filter_cell),
        "kept_cells": str(x.shape[0]),
        "removed_cells": str(removed),
        "removed_pct": f"{removed / raw_cells:.2%}" if raw_cells else "NA",
        "pre_gene_detail": pre_gene_detail,
        "cell_step_detail": "; ".join(cell_details),
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-dir", default="examples/tuning", type=Path)
    parser.add_argument("--server", default="112")
    parser.add_argument("--method", help="Only report one CSV method name, e.g. Graphsc")
    parser.add_argument("--dataset", help="Only report one normalized dataset name, e.g. mouse_kidney_drop")
    parser.add_argument("--output", type=Path, help="Write the full CSV result to this path")
    parser.add_argument("--summary-file", type=Path, help="Existing step2 summary CSV to annotate with cell counts")
    parser.add_argument("--summary-output", type=Path, help="Where to save the annotated step2 summary CSV")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if args.summary_file:
        if not args.method or not args.dataset:
            raise ValueError("--summary-file requires --method and --dataset")
        output = args.summary_output or args.summary_file.with_name(f"{args.summary_file.stem}_with_cell_counts.csv")
        annotate_summary(args.tuning_dir, args.method, args.dataset, args.summary_file, output,
                         download=not args.no_download)
        print(output)
        return

    rows = []
    for method, dataset, config in server_112_configs(args.tuning_dir, args.server):
        if args.method and method != args.method:
            continue
        if args.dataset and dataset != args.dataset:
            continue
        rows.extend(summarize_one(args.tuning_dir, method, dataset, config, download=not args.no_download))
    fields = [
        "method",
        "dataset",
        "scenario",
        "gene_filter",
        "cell_filter",
        "raw_cells",
        "raw_genes",
        "cells_before_filter_cell",
        "genes_before_filter_cell",
        "kept_cells",
        "removed_cells",
        "removed_pct",
        "pre_gene_detail",
        "cell_step_detail",
        "h5",
        "yaml",
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
