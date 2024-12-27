import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import os_sort_key
from omegaconf import OmegaConf
from sympy import im
from tqdm import tqdm

from dance.settings import METADIR
from dance.utils import try_import

# get yaml of best method


def check_identical_strings(string_list):
    """Compare strings in a list to check if they are identical.

    Parameters
    ----------
    string_list : list
        List of strings to compare

    Returns
    -------
    str
        The common string if all strings are identical

    Raises
    ------
    ValueError
        If list is empty or strings are different

    """
    if not string_list:
        raise ValueError("The list is empty")

    arr = np.array(string_list)
    if not np.all(arr == arr[0]):
        raise ValueError("Different strings found")

    return string_list[0]

    # if not string_list:
    #     raise ValueError("The list is empty")
    # first_string = string_list[0]
    # for s in string_list[1:]:
    #     if s != first_string:
    #         raise ValueError(f"Different strings found: '{first_string}' and '{s}'")
    # return first_string


def get_sweep_url(step_csv: pd.DataFrame, single=True):
    """Extract Weights & Biases sweep URL from a DataFrame containing run IDs.

    Parameters
    ----------
    step_csv : pd.DataFrame
        DataFrame containing wandb run IDs in an 'id' column
    single : bool, optional
        If True, only process the first run, by default True

    Returns
    -------
    str
        The wandb sweep URL

    """
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
    """Parse Weights & Biases URL to extract entity, project and sweep components.

    Parameters
    ----------
    url : str
        Complete wandb sweep URL

    Returns
    -------
    tuple or None
        Tuple of (entity, project, sweep_id) if parsing succeeds
        None if parsing fails

    """
    pattern = r"https://wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/]+)"

    match = re.search(pattern, url)

    if match:
        entity = match.group(1)
        project = match.group(2)
        pattern = r'/sweeps/([^/?]+)'  # Regular expression pattern
        match = re.search(pattern, url)
        if match:
            sweep_id = match.group(1)
            return entity, project, sweep_id
        return None
    else:
        print(url)
        print("No match found")


def get_best_method(urls, metric_col="test_acc"):
    """Find the best performing method across multiple wandb sweeps.

    Parameters
    ----------
    urls : list
        List of wandb sweep URLs to compare
    metric_col : str, optional
        Metric column name to use for comparison, by default "test_acc"

    Returns
    -------
    tuple
        (best_step_name, best_run, best_metric_value) where:
        - best_step_name: name of the step with best performance
        - best_run: wandb run object of best performing run
        - best_metric_value: value of the metric for best run

    """
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
    """Generate YAML configuration for the best performing wandb run.

    Parameters
    ----------
    step_name : str
        Name of the step ('step2' or 'step3_X')
    best_run : wandb.Run
        Best performing wandb run object
    file_path : str
        Path to configuration files

    Returns
    -------
    str
        YAML string containing the best configuration

    """
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
    """Check if results directory exists and contains multiple result files.

    Parameters
    ----------
    file_path : str
        Path to check for results

    Returns
    -------
    bool
        True if valid results exist (directory exists and contains >1 file)

    """
    file_path = f"{file_path}/results/params/"
    if os.path.exists(file_path) and os.path.isdir(file_path):
        file_num = len(os.listdir(file_path))
        return file_num > 1
    else:
        return False


def get_new_ans(tissue):
    ans = []
    # temp=all_datasets[all_datasets["tissue"] == tissue]["data_fname"].tolist()
    collect_datasets = [
        collect_dataset.split(tissue)[1].split("_")[0]
        for collect_dataset in all_datasets[all_datasets["tissue"] == tissue]["data_fname"].tolist()
    ]

    for method_folder in tqdm(methods):
        for dataset_id in collect_datasets:
            file_path = f"../tuning/{method_folder}/{dataset_id}"
            if not check_exist(file_path):
                continue
            step2_url = get_sweep_url(pd.read_csv(f"{file_path}/results/pipeline/best_test_acc.csv"))
            step3_urls = []
            for i in range(3):
                file_csv = f"{file_path}/results/params/{i}_best_test_acc.csv"
                if not os.path.exists(file_csv):  # no parameter
                    print(f"File {file_csv} does not exist, skipping.")
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
    new_df = pd.DataFrame(ans)
    return new_df


def write_ans(tissue, new_df, output_file=None):
    """Process and write results for a specific tissue type to CSV."""
    if output_file is None:
        output_file = f"sweep_results/{tissue}_ans.csv"
    
    # 确保Dataset_id是索引
    if 'Dataset_id' in new_df.columns:
        new_df = new_df.set_index('Dataset_id')
    
    # 处理新数据，合并相同Dataset_id的非NA值
    new_df_processed = pd.DataFrame()
    for idx in new_df.index.unique():
        row_data = {}
        subset = new_df.loc[new_df.index == idx]
        for col in new_df.columns:
            values = subset[col].dropna().unique()
            if len(values) > 0:
                row_data[col] = values[0]
        new_df_processed = pd.concat([
            new_df_processed, 
            pd.DataFrame(row_data, index=[idx])
        ])
    
    if os.path.exists(output_file):
        # 读取现有数据
        existing_df = pd.read_csv(output_file)
        if 'Dataset_id' in existing_df.columns:
            existing_df = existing_df.set_index('Dataset_id')
        
        # 创建合并后的DataFrame，包含所有列
        merged_df = existing_df.copy()
        # 添加新数据中的列（如果不存在）
        for col in new_df_processed.columns:
            if col not in merged_df.columns:
                merged_df[col] = pd.NA
        # 对每个Dataset_id进行合并和冲突检查
        for idx in new_df_processed.index:
            if idx in existing_df.index:
                # 检查每一列的值
                for col in new_df_processed.columns:
                    new_value = new_df_processed.loc[idx, col]
                    # 检查列是否存在于现有数据中
                    if col in existing_df.columns:
                        existing_value = existing_df.loc[idx, col]
                        # 只对_best_res结尾的列进行冲突检查
                        if str(col).endswith("_best_res"):
                            if pd.notna(new_value) and pd.notna(existing_value):
                                if abs(new_value - existing_value) > 1e-10:
                                    raise ValueError(f"结果冲突: Dataset {idx}, Column {col}\n"
                                                  f"现有值: {existing_value}\n新值: {new_value}")
                                else:
                                    print(f"提示: 发现重复值 Dataset {idx}, Column {col}\n"
                                        f"现有值和新值都是: {new_value}")
                    # 如果新值不是NaN，更新该值
                    if pd.notna(new_value):
                        merged_df.loc[idx, col] = new_value
            else:
                # 如果是新的Dataset_id，直接添加整行
                merged_df.loc[idx] = new_df_processed.loc[idx]
        
        # 保存合并后的数据
        merged_df.to_csv(output_file)
    else:
        # 如果文件不存在，直接保存处理后的新数据
        new_df_processed.to_csv(output_file)


wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
if __name__ == "__main__":
    # Initialize wandb and set global configuration
    # Load dataset configuration and process results for tissue
    all_datasets = pd.read_csv(METADIR / "scdeepsort.csv", header=0, skiprows=[i for i in range(1, 69)])
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", type=str, default="Heart")
    args = parser.parse_args()
    tissue = args.tissue.capitalize()
    new_df = get_new_ans(tissue)
    write_ans(tissue, new_df)
