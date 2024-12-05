import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import os_sort_key
from omegaconf import OmegaConf
from sympy import im
from tqdm import tqdm

from dance.utils import try_import

# get yaml of best method


def check_identical_strings(string_list):
    if not string_list:
        raise ValueError("列表为空")

    arr = np.array(string_list)
    if not np.all(arr == arr[0]):
        raise ValueError("发现不同的字符串")

    return string_list[0]


    # if not string_list:
    #     raise ValueError("列表为空")
    # first_string = string_list[0]
    # for s in string_list[1:]:
    #     if s != first_string:
    #         raise ValueError(f"发现不同的字符串: '{first_string}' 和 '{s}'")
    # return first_string
def get_sweep_url(step_csv: pd.DataFrame, single=True):
    ids = step_csv["id"]
    sweep_urls = []
    for run_id in tqdm(reversed(ids),
                       leave=False):  #The reversal of order is related to additional_sweep_ids.append(sweep_id)
        api = wandb.Api()
        run = api.run(f"/{entity}/{project}/runs/{run_id}")
        sweep_urls.append(run.sweep.url)
        if single:
            break
    sweep_url = check_identical_strings(sweep_urls)
    return sweep_url


import re


def spilt_web(url: str):
    pattern = r"https://wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/]+)"

    match = re.search(pattern, url)

    if match:
        entity = match.group(1)
        project = match.group(2)
        sweep_id = match.group(3)

        return entity, project, sweep_id
    else:
        print(url)
        print("No match found")


def get_best_method(urls, metric_col="test_acc"):
    all_best_run = None
    all_best_step_name = None
    step_names = ["step2", "step3_0", "step3_1", "step3_2"]

    def get_metric(run):
        if metric_col not in run.summary:
            return float('-inf')
        else:
            return run.summary[metric_col]

    for step_name, url in zip(step_names, urls):
        _, _, sweep_id = spilt_web(url)
        sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
        goal = sweep.config["metric"]["goal"]
        if goal == "maximize":
            best_run = max(sweep.runs, key=get_metric)
        elif goal == "minimize":
            best_run = min(sweep.runs, key=get_metric)
        else:
            raise RuntimeError("choose goal in ['minimize','maximize']")
        if metric_col not in best_run.summary:
            continue
        if all_best_run is None:
            all_best_run = best_run
            all_best_step_name = step_name
        elif all_best_run.summary[metric_col] < best_run.summary[metric_col] and goal == "maximize":
            all_best_run = best_run
            all_best_step_name = step_name
        elif all_best_run.summary[metric_col] > best_run.summary[metric_col] and goal == "minimize":
            all_best_run = best_run
            all_best_step_name = step_name
    return all_best_step_name, all_best_run, all_best_run.summary[metric_col]


def get_best_yaml(step_name, best_run, file_path):
    if step_name == "step2":
        conf = OmegaConf.load(f"{file_path}/pipeline_params_tuning_config.yaml")
        for i, fun in enumerate(conf["pipeline"]):
            if "include" not in fun:
                continue
            type_fun = fun["type"]
            prefix = f"pipeline.{i}.{type_fun}"
            # filtered_dict = {k: v for k, v in b_run.config.items() if k==prefix}.items()[0]
            fun_name = best_run.config[prefix]
            fun['target'] = fun_name
            if 'params' not in fun:
                fun['params'] = {}
            if "default_params" in fun and fun_name in fun["default_params"]:
                fun['params'].update(fun["default_params"][fun_name])
            del fun["include"]
            del fun["default_params"]
    else:
        step3_number = step_name.split("_")[1]
        conf = OmegaConf.load(f"{file_path}/config_yamls/params/{step3_number}_test_acc_params_tuning_config.yaml")
        for i, fun in enumerate(conf['pipeline']):
            if 'params_to_tune' not in fun:
                continue
            target = fun["target"]
            prefix = f"params.{i}.{target}"
            filtered_dict = {k: v for k, v in best_run.config.items() if k.startswith(prefix)}
            for k, v in filtered_dict.items():
                param_name = k.split(".")[-1]
                fun['params_to_tune'][param_name] = v
            if "params" not in fun:
                fun["params"] = {}
            fun["params"].update(fun['params_to_tune'])
            del fun["params_to_tune"]
    return OmegaConf.to_yaml(conf["pipeline"])


def check_exist(file_path):
    file_path = f"{file_path}/results/params/"
    if os.path.exists(file_path) and os.path.isdir(file_path):
        file_num = len(os.listdir(file_path))
        return file_num > 1
    else:
        return False


def write_ans(tissue):
    ans = []
    collect_datasets = all_datasets[tissue]

    for method_folder in tqdm(collect_datasets):
        for dataset_id in collect_datasets[method_folder]:
            file_path = f"{file_root}/{method_folder}/{dataset_id}"
            if not check_exist(file_path):
                continue
            step2_url = get_sweep_url(pd.read_csv(f"{file_path}/results/pipeline/best_test_acc.csv"))
            step3_urls = []
            for i in range(3):
                file_csv = f"{file_path}/results/params/{i}_best_test_acc.csv"
                if not os.path.exists(file_csv):  #no parameter
                    print(f"文件 {file_csv} 不存在，跳过。")
                    continue
                step3_urls.append(get_sweep_url(pd.read_csv(file_csv)))
            step3_str = ",".join(step3_urls)
            step_str = f"step2:{step2_url}|step3:{step3_str}"
            step_name, best_run, best_res = get_best_method([step2_url] + step3_urls)
            best_yaml = get_best_yaml(step_name, best_run, file_path)
            ans.append({
                "Dataset_id": dataset_id,
                method_folder: step_str,
                f"{method_folder}_best_yaml": best_yaml,
                f"{method_folder}_best_res": best_res
            })
    # with open('temp_ans.json', 'w') as f:
    #     json.dump(ans, f,indent=4)
    pd.DataFrame(ans).to_csv(f"{tissue}_ans.csv")


if __name__ == "__main__":
    wandb = try_import("wandb")
    entity = "xzy11632"
    project = "dance-dev"
    file_root = str(Path(__file__).resolve().parent)
    with open(f"{file_root}/dataset_server.json") as f:
        all_datasets = json.load(f)
    file_root = "./tuning"
    tissues = ["heart"]
    for tissue in tissues:
        write_ans(tissue)
