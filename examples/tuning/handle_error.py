import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import wandb

entity = "xzy11632"
project = "dance-dev"


def read_log(file_path="/home/zyxing/dance/examples/tuning/cluster_graphsc/mouse_kidney_cell/out.log", sweep_id=None):
    run_ids = []
    if sweep_id is not None:
        sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
        for run in sweep.runs:
            run_ids.append(run.id)
    with open(file_path) as file:
        lines = file.readlines()
    runs = []
    current_run = []
    start_index = 0
    for index, line in enumerate(lines):
        if "wandb: Agent Starting Run:" in line:
            start_index = index
            break
    for line in lines[start_index:]:
        if "wandb: Agent Starting Run:" in line:
            if current_run:
                runs.append(current_run)
                current_run = []
        current_run.append(line.strip())
    if current_run:
        runs.append(current_run)
    err_data = []
    for run in runs:
        pip_dict = get_pip_dict(run, run_ids)
        if pip_dict is not None:
            err_data.append(pip_dict)
    grouped_dicts = defaultdict(list)
    for d in err_data:
        # print(frozenset(list(d.keys())))
        if "sweep_id" in d.keys():
            grouped_dicts[d["sweep_id"]].append(d)
    dataframes = []
    sweep_ids = []
    for group_key, group_dicts in grouped_dicts.items():
        df = pd.DataFrame(group_dicts)
        sweep_ids.append(group_key)
        dataframes.append(df)
    for sweep_id, df in zip(sweep_ids, dataframes):
        save_path = Path(Path(file_path).parent, f"{sweep_id}_err.csv").resolve()
        df.to_csv(save_path)


def get_run_id(text):
    pattern = r'Run: (\w+) with'
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
        return result.strip()
    else:
        print("No match found.")


def get_pip_dict(run_lines, run_ids):
    run_id = get_run_id(run_lines[0])
    pip_dict = {}
    err_lines = []
    pip_prefix = "wandb: 	pipeline"
    param_prefix = "wandb: 	params"
    error_prefix = "wandb: ERROR "
    sweep_prefix = f"wandb: 🧹 View sweep at https://wandb.ai/{entity}/{project}/sweeps/"
    for line in run_lines:
        if line.startswith(param_prefix) or line.startswith(pip_prefix):
            _, name, key = line.split(":", 2)
            name, key = name.strip(), key.strip()
            pip_dict[name] = key
        if line.startswith(error_prefix):
            err_lines.append(line)
        if line.startswith(sweep_prefix):
            pip_dict["sweep_id"] = line[len(sweep_prefix):]

    if len(err_lines) == 0 or (len(run_ids) > 0 and run_id not in run_ids):
        return None
    if len(run_ids) == 0 or run_id in run_ids:
        pip_dict["info"] = "\n".join(err_lines)
        pip_dict["run_id"] = run_id
    return pip_dict


def list_files(directory):
    path = Path(directory)
    for file_path in path.rglob('*'):
        if file_path.is_file():
            if file_path.name == "out.log":
                read_log(file_path)


if __name__ == "__main__":
    list_files("/home/zyxing/dance/examples/tuning")
