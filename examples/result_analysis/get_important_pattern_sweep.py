import argparse
import json
import sys
from pathlib import Path
from turtle import pos

import pandas as pd
import requests
from get_important_pattern import get_com_all, get_forest_model_pattern, get_frequent_itemsets
from numpy import choose

sys.path.append("..")
from get_result_web import spilt_web

from dance.pipeline import flatten_dict
from dance.utils import try_import

entity = "xzy11632"
project = "dance-dev"
tasks = ["cell type annotation new", "clustering", "imputation_new", "spatial domain", "cell type deconvolution"]
mertic_names = ["test_acc", "acc", "MRE", "ARI", "MSE"]
ascendings = [False, False, True, False, True]

multi_mod = False
if multi_mod:
    raise NotImplementedError("multi mod")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--positive", action='store_true')
parser.add_argument("--only_apr", action='store_true')
parser.add_argument("--choose_tasks", nargs="+", default=tasks)
args = parser.parse_args()
choose_tasks = args.choose_tasks
positive = args.positive
only_apr = args.only_apr
if not positive:
    assert only_apr
    ascendings = [not item for item in ascendings]
file_root = Path(__file__).resolve().parent
prefix = f'https://wandb.ai/{entity}/{project}'
runs_sum = 0
wandb = try_import("wandb")


def get_additional_sweep(sweep_id):
    # if sweep has piror runs
    # every run get command , get additional sweep id
    # or last run command
    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
    #last run command
    run = next((t_run for t_run in sweep.runs if t_run.state == "finished"), None)
    additional_sweep_ids = [sweep_id]
    if run is None:  # check summary data count, note aznph5wt, quantities may be inconsistent
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
    columns = sorted([col for col in step2_origin_data.columns if col.startswith("pipeline")])
    step2_data = step2_origin_data.loc[:, columns + [metric_name]]
    # com_ans = get_com_all(step2_data, metric_name, ascending, vis=vis, alpha=alpha)
    step2_data[metric_name] = step2_data[metric_name].astype(float)
    if not ascending:
        min_metric = step2_data[metric_name].min()
        if pd.isna(min_metric):
            return {
                "error":
                f"All {metric_name} values are NaN and the minimum cannot be calculated. Please check your data."
            }
        step2_data[metric_name] = step2_data[metric_name].fillna(0)  #if ascending=False
    else:
        max_metric = step2_data[metric_name].max()
        if pd.isna(max_metric):
            return {
                "error":
                f"All {metric_name} values are NaN and the maximum cannot be calculated. Please check your data."
            }
        print(f"\nmax {metric_name}:{max_metric}")
        buffer_percentage = 0.2  # 20%
        replacement = max_metric * (1 + buffer_percentage)
        step2_data[metric_name] = step2_data[metric_name].fillna(replacement)
    apr_ans = get_frequent_itemsets(step2_data, metric_name, ascending)
    if positive and not only_apr:
        return {"forest_model": get_forest_model_pattern(step2_data, metric_name), "apr_ans": apr_ans}
    else:
        return {"apr_ans": apr_ans}
    # except Exception as e:
    #     print(e)
    #     return str(e)


if __name__ == "__main__":
    start = True
    ans_all = []
    for i, task in enumerate(tasks):

        if task not in choose_tasks:
            continue
        data = pd.read_excel(file_root / "results.xlsx", sheet_name=task, dtype=str)
        data = data.ffill().set_index(['Methods'])
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):

                method = data.index[row_idx]
                dataset = data.columns[col_idx]
                value = data.iloc[row_idx, col_idx]
                step_name = data.iloc[row_idx]["Unnamed: 1"]
                # if dataset=="Dataset6:pancreatic_cancer" and method == "Stlearn":
                #     start=True
                if not start:
                    continue
                # if method !="ACTINN" :
                #     continue
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
                ans_single = {
                    "task": task,
                    "dataset": dataset,
                    "method": method,
                    "pattern": summary_pattern(ans, mertic_names[i], ascendings[i])
                }
                with open(
                        f"dance_auto_preprocess/patterns/{'only_apr_' if only_apr else ''}{'neg_' if not positive else ''}{task}_{dataset}_{method}_pattern.json",
                        "w") as f:
                    json.dump(ans_single, f, indent=2)
                ans_all.append(ans_single)
                print(dataset)
                print(method)
    with open(f"pattern.json", "w") as f:
        json.dump(ans_all, f, indent=2)
