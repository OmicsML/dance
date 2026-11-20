#!/usr/bin/env python
"""Generate commands to rebuild cluster step3 sweeps on one server.

The generated script is intended for the case where step2 sweeps already exist:
it can skip running new step2 agents, download existing step2 summaries, apply
cell-count guards when selecting the top pipelines, and then run step3.

"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ClusterTask:
    method: str
    method_dir: str
    dataset: str
    sweep_id: str
    yaml_path: Path


def dataset_dir(name: str) -> str:
    return DATASET_DIR.get(name, name.replace(" ", "_"))


def sweep_id_from_url(value: str) -> str | None:
    match = re.search(r"/sweeps/([A-Za-z0-9_-]+)", value)
    return match.group(1) if match else None


def server_tasks(tuning_dir: Path, server: str) -> list[ClusterTask]:
    csv_path = tuning_dir / "results - cluster.csv"
    rows = list(csv.reader(csv_path.open(newline="")))
    header = rows[0]
    server_re = re.compile(r"^(?:\d+|papermachine)$")
    url_re = re.compile(r"https?://")

    current_method = None
    step12_row: list[str] | None = None
    tasks: list[ClusterTask] = []
    for row in rows[1:]:
        row = row + [""] * (len(header) - len(row))
        method = row[0].strip()
        step_name = row[1].strip()
        if method:
            current_method = method
            step12_row = None
        if current_method not in METHOD_DIR:
            continue
        if step_name == "Step 1&2":
            step12_row = row
            continue

        values = [cell.strip() for cell in row[2:]]
        is_server_row = (step12_row is not None and any(server_re.fullmatch(value) for value in values)
                         and not any(url_re.search(value) for value in values))
        if not is_server_row:
            continue

        for col_idx, value in enumerate(values, start=2):
            if value != server:
                continue
            dataset = dataset_dir(header[col_idx])
            sweep_id = sweep_id_from_url(step12_row[col_idx])
            if sweep_id is None:
                continue
            method_dir = METHOD_DIR[current_method]
            yaml_path = tuning_dir / method_dir / dataset / "pipeline_params_tuning_config.yaml"
            if yaml_path.exists():
                tasks.append(ClusterTask(current_method, method_dir, dataset, sweep_id, yaml_path))
    return tasks


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def archive_lines(task: ClusterTask, tuning_dir: Path) -> list[str]:
    task_rel = Path(task.method_dir) / task.dataset
    archive_rel = Path("${archive_root}") / task.method_dir / task.dataset
    return [
        f"mkdir -p {archive_rel}",
        f"if [ -d {task_rel / 'results'} ]; then mv {task_rel / 'results'} {archive_rel}/results; fi",
        f"if [ -d {task_rel / 'config_yamls' / 'params'} ]; then mkdir -p {archive_rel}/config_yamls; "
        f"mv {task_rel / 'config_yamls' / 'params'} {archive_rel}/config_yamls/params; fi",
        f"if [ -d {task_rel / 'temp_data' / 'results'} ]; then mkdir -p {archive_rel}/temp_data; "
        f"mv {task_rel / 'temp_data' / 'results'} {archive_rel}/temp_data/results; fi",
        f"if [ -d {task_rel / 'temp_data' / 'config_yamls' / 'params'} ]; then "
        f"mkdir -p {archive_rel}/temp_data/config_yamls; "
        f"mv {task_rel / 'temp_data' / 'config_yamls' / 'params'} {archive_rel}/temp_data/config_yamls/params; fi",
        f"if [ -f {task_rel / 'out.log'} ]; then mv {task_rel / 'out.log'} {archive_rel}/out.log; fi",
        f"if [ -f {task_rel / 'temp_data' / 'out.log'} ]; then mkdir -p {archive_rel}/temp_data; "
        f"mv {task_rel / 'temp_data' / 'out.log'} {archive_rel}/temp_data/out.log; fi",
        f"if [ -f {task_rel / 'out_step3_cell_guard.log'} ]; then "
        f"mv {task_rel / 'out_step3_cell_guard.log'} {archive_rel}/out_step3_cell_guard.log; fi",
    ]


def command_line(task: ClusterTask, args) -> str:
    cmd = [
        "CUDA_VISIBLE_DEVICES=2",
        "python",
        "main.py",
        "--dataset",
        task.dataset,
        "--count",
        str(args.count),
        "--device",
        args.device,
        "--sweep_id",
        task.sweep_id,
        "--cell_count_summary_path",
        "../cluster_cell_filtering_112.csv",
        "--min_cell_retention",
        str(args.min_cell_retention),
    ]
    if args.min_cell_count is not None:
        cmd += ["--min_cell_count", str(args.min_cell_count)]
    if args.auto_additional:
        cmd.append("--auto_additional_sweep_ids")
    redirect = f">> {shlex.quote(task.dataset + '/out_step3_cell_guard.log')} 2>&1"
    return f"(cd {shlex.quote(task.method_dir)} && {shell_join(cmd)} {redirect})"


def render_script(tasks: list[ClusterTask], tuning_dir: Path, args) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(tuning_dir.resolve()))}",
        "export NUMBA_CACHE_DIR=/tmp/numba-cache-${USER:-zyxing}",
        "export NUMBA_DISABLE_CACHE=1",
        "export MPLCONFIGDIR=/tmp/mpl-cache-${USER:-zyxing}",
        "mkdir -p ${NUMBA_CACHE_DIR} ${MPLCONFIGDIR}",
        'archive_root="cluster_112_step3_archive/$(date +%Y%m%d_%H%M%S)"',
        "mkdir -p ${archive_root}",
        "",
    ]
    for task in tasks:
        lines.append(f"# {task.method} / {task.dataset} / step2 sweep {task.sweep_id}")
        if args.archive:
            lines.extend(archive_lines(task, tuning_dir))
        line = command_line(task, args)
        lines.append(f"echo '[START] {task.method}/{task.dataset}'")
        if args.background:
            lines.append(f"({line} || echo '[FAILED] {task.method}/{task.dataset}') &")
        else:
            lines.append(f"if ! {line}; then echo '[FAILED] {task.method}/{task.dataset}'; fi")
        lines.append("")
    if args.background:
        lines.append("wait")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-dir", default="examples/tuning", type=Path)
    parser.add_argument("--server", default="112")
    parser.add_argument("--output", default="examples/tuning/run_cluster_112_step3_cell_guard.sh", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", default=0, type=int,
                        help="Use 0 to skip step2 agents and only reuse existing sweep summaries")
    parser.add_argument("--min-cell-retention", default=0.8, type=float)
    parser.add_argument("--min-cell-count", default=None, type=int)
    parser.add_argument("--no-auto-additional", dest="auto_additional", action="store_false",
                        help="Do not add --auto_additional_sweep_ids to generated commands")
    parser.add_argument("--no-archive", dest="archive", action="store_false")
    parser.add_argument("--background", action="store_true", help="Run generated task commands in the background")
    parser.set_defaults(auto_additional=True, archive=True)
    args = parser.parse_args()

    tasks = server_tasks(args.tuning_dir, args.server)
    script = render_script(tasks, args.tuning_dir, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script + "\n")
    print(f"Wrote {args.output} with {len(tasks)} tasks")


if __name__ == "__main__":
    main()
