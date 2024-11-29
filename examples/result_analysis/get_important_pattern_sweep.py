import json
import sys
from pathlib import Path
from turtle import pos

import pandas as pd
import requests
from get_important_pattern import get_com_all, get_frequent_itemsets

sys.path.append("..")
from get_result_web import spilt_web

from dance.pipeline import flatten_dict
from dance.utils import try_import

entity = "xzy11632"
project = "dance-dev"
tasks = ["cell type annotation new", "clustering", "imputation_new", "spatial domain", "cell type deconvolution"]
mertic_names = ["test_acc", "acc", "MRE", "ARI", "MSE"]
ascendings = [False, False, True, False, True]
file_root = Path(__file__).resolve().parent
prefix = f'https://wandb.ai/{entity}/{project}'
runs_sum = 0
wandb = try_import("wandb")
positive = True


def get_additional_sweep(sweep_id):
    # if sweep has piror runs
    # every run get command , get additional sweep id
    # or last run command
    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
    #last run command
    run = next((t_run for t_run in sweep.runs if t_run.state == "finished"), None)
    additional_sweep_ids = [sweep_id]
    if run is None:  #check summary data num,note aznph5wt,数量可能不一致。
        return additional_sweep_ids
    run_id = run.id
    web_abs = requests.get(f"https://api.wandb.ai/files/{run.entity}/{run.project}/{run_id}/wandb-metadata.json")
    args = dict(web_abs.json())["args"]
    for i in range(len(args)):
        if args[i] == '--additional_sweep_ids':
            if i + 1 < len(args):
                additional_sweep_ids += get_additional_sweep(args[i + 1])
    return additional_sweep_ids


def summary_pattern(step2_origin_data, metric_name, ascending, alpha=0.05, vis=False):
    # try:
    step2_data = step2_origin_data.dropna()
    com_ans = get_com_all(step2_data, metric_name, ascending, vis=vis, alpha=alpha)
    apr_ans = get_frequent_itemsets(step2_data, metric_name, ascending)
    return list(set(com_ans) & set(apr_ans))
    # except Exception as e:
    #     print(e)
    #     return str(e)


if __name__ == "__main__":
    ans_all = []
    for i, task in enumerate(tasks):
        data = pd.read_excel(file_root / "results.xlsx", sheet_name=task, dtype=str)
        data = data.ffill().set_index(['Methods'])
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                method = data.index[row_idx]
                dataset = data.columns[col_idx]
                value = data.iloc[row_idx, col_idx]
                step_name = data.iloc[row_idx]["Unnamed: 1"]
                if method != "SVM" or dataset != "Dataset 1: GSE67835 Brain":
                    continue
                if isinstance(value, str) and value.startswith(prefix) and (
                        str(step_name).lower() == "step2" or str(step_name).lower() == "step 2"):  #TODO add step3
                    sweep_url = value
                else:
                    continue
                _, _, sweep_id = spilt_web(sweep_url)
                sweep_ids = get_additional_sweep(sweep_id)
                summary_data = []
                for sweep_id in sweep_ids:
                    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
                    for run in sweep.runs:
                        result = dict(run.summary._json_dict).copy()
                        result.update(run.config)
                        result.update({"id": run.id})
                        summary_data.append(flatten_dict(result))  # get result and config
                ans = pd.DataFrame(summary_data).set_index(["id"])
                ans.sort_index(axis=1, inplace=True)
                print(dataset)
                print(method)
                ans_all.append({
                    "task": task,
                    "dataset": dataset,
                    "method": method,
                    "pattern": summary_pattern(ans, mertic_names[i], ascendings[i])
                })
    with open(f"positive:{positive}_pattern.json", "w") as f:
        json.dump(ans_all, f, indent=2)
