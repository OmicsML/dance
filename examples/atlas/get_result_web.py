import argparse
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import os_sort_key
from omegaconf import OmegaConf
from sympy import im
from tqdm import tqdm

from dance import logger
from dance.settings import DANCEDIR, METADIR
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
    test_accs_exists = False
    if "test_acc" in step_csv.columns:
        test_accs = reversed(step_csv["test_acc"])
        test_accs_exists = True
    else:
        test_accs = [np.nan] * len(ids)

    for run_id, test_acc in tqdm(
            zip(reversed(ids),
                test_accs), leave=False):  #The reversal of order is related to additional_sweep_ids.append(sweep_id)
        if test_accs_exists and pd.isna(test_acc):
            continue
        api = wandb.Api(timeout=1000)
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


def get_metric(run, metric_col):
    """Extract metric value from wandb run.

    Parameters
    ----------
    run : wandb.Run
        Weights & Biases run object

    Returns
    -------
    float
        Metric value or negative infinity if metric not found

    """
    if metric_col not in run.summary:
        return float('-inf')  # Return -inf for missing metrics to handle in comparisons
    return run.summary[metric_col]


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
    step2_best_run = None
    # Track run statistics
    run_states = {"all_total_runs": 0, "all_finished_runs": 0}

    for step_name, url in zip(step_names, urls):
        _, _, sweep_id = spilt_web(url)
        sweep = wandb.Api(timeout=1000).sweep(f"{entity}/{project}/{sweep_id}")

        # Update run statistics
        finished_runs = [run for run in sweep.runs if run.state == "finished"]
        run_states.update({
            f"{step_name}_total_runs": len(sweep.runs),
            f"{step_name}_finished_runs": len(finished_runs)
        })
        run_states["all_total_runs"] += run_states[f"{step_name}_total_runs"]
        run_states["all_finished_runs"] += run_states[f"{step_name}_finished_runs"]

        # Find best run based on optimization goal
        goal = sweep.config["metric"]["goal"]
        best_run = max(sweep.runs, key=partial(get_metric, metric_col=metric_col)) if goal == "maximize" else \
                   min(sweep.runs, key=partial(get_metric, metric_col=metric_col)) if goal == "minimize" else \
                   None

        if best_run is None:
            raise RuntimeError("Optimization goal must be either 'minimize' or 'maximize'")

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
        if step2_best_run is None and step_name == "step2":
            step2_best_run = best_run
    num = run_states["all_finished_runs"] / run_states["all_total_runs"]
    run_states["finished_rate"] = f"{num:.2%}"
    need_to_check = num < 0.6
    runs_states_str = "|".join([f"{k}:{v}" for k, v in run_states.items()])
    return all_best_step_name, all_best_run, all_best_run.summary[
        metric_col], runs_states_str, need_to_check, step2_best_run, step2_best_run.summary[metric_col]


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
        (collect_dataset.split(tissue)[1] +
         (tissue + collect_dataset.split(tissue)[2] if len(collect_dataset.split(tissue)) >= 3 else '')).split('_')[0]
        for collect_dataset in all_datasets[all_datasets["tissue"] == tissue]["data_fname"].tolist()
    ]

    for method_folder in tqdm(methods):
        for dataset_id in collect_datasets:
            if dataset_id == "0b4a15a7-4e9e-4555-9733-2423e5c66469":  #f72958f5-7f42-4ebb-98da-445b0c6de516
                pass
            file_path = DANCEDIR / f"examples/tuning/{method_folder}/{dataset_id}"
            if not check_exist(file_path):
                continue
            step2_url = get_sweep_url(pd.read_csv(f"{file_path}/results/pipeline/best_test_acc.csv"))
            step3_urls = []
            for i in range(3):
                file_csv = f"{file_path}/results/params/{i}_best_test_acc.csv"
                if not os.path.exists(file_csv):
                    print(f"File {file_csv} does not exist, skipping.")
                    continue
                step3_urls.append(get_sweep_url(pd.read_csv(file_csv)))
            step3_str = ",".join(step3_urls)
            step_str = f"step2:{step2_url}|step3:{step3_str}"
            step_name, best_run, best_res, run_stats_str, need_to_check, step2_best_run, step2_best_res = get_best_method(
                [step2_url] + step3_urls)
            best_yaml = get_best_yaml(step_name, best_run, file_path)
            step2_best_yaml = get_best_yaml("step2", step2_best_run, file_path)
            ans.append({
                "Dataset_id": dataset_id,
                method_folder: step_str,
                f"{method_folder}_best_yaml": best_yaml,
                f"{method_folder}_best_res": best_res,
                f"{method_folder}_run_stats": run_stats_str,
                f"{method_folder}_check": need_to_check,
                f"{method_folder}_step2_best_yaml": step2_best_yaml,
                f"{method_folder}_step2_best_res": step2_best_res
            })
    # with open('temp_ans.json', 'w') as f:
    #     json.dump(ans, f,indent=4)
    new_df = pd.DataFrame(ans)
    return new_df


def write_ans(tissue, new_df, output_file=None):
    """Process and write results for a specific tissue type to CSV.

    Updates all columns with matching method_folder prefix only when new _best_res
    value is greater than existing value.

    Parameters
    ----------
    tissue : str
        Tissue type being processed
    new_df : pd.DataFrame
        New results to be written
    output_file : str, optional
        Output file path. Defaults to 'sweep_results/{tissue}_ans.csv'

    """
    if output_file is None:
        output_file = f"sweep_results/{tissue}_ans.csv"

    if 'Dataset_id' not in new_df.columns:
        logger.warning("Dataset_id column missing in input DataFrame")
        return

    # Reset index to ensure Dataset_id is a regular column
    new_df = new_df.reset_index(drop=True)

    # Process new data by merging rows with same Dataset_id
    new_df_processed = pd.DataFrame()
    for dataset_id in new_df['Dataset_id'].unique():
        row_data = {'Dataset_id': dataset_id}
        subset = new_df[new_df['Dataset_id'] == dataset_id]
        for col in new_df.columns:
            if col != 'Dataset_id':
                non_null_values = subset[col].dropna().unique()
                if len(non_null_values) > 0:
                    row_data[col] = non_null_values[0]
        new_df_processed = pd.concat([new_df_processed, pd.DataFrame([row_data])])

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_df = existing_df.loc[:, ~existing_df.columns.str.contains('^Unnamed')]
        merged_df = existing_df.copy()

        for col in new_df_processed.columns:
            if col not in merged_df.columns:
                merged_df[col] = pd.NA

        # 遍历每个新数据行
        for _, new_row in new_df_processed.iterrows():
            dataset_id = new_row['Dataset_id']
            existing_row = merged_df[merged_df['Dataset_id'] == dataset_id]

            if len(existing_row) > 0:
                # 检查每个method的best_res
                for method in methods:
                    best_res_col = f"{method}_best_res"
                    if best_res_col in new_row and best_res_col in existing_row:
                        new_value = new_row[best_res_col]
                        existing_value = existing_row[best_res_col].iloc[0]

                        # 只有当新值存在且大于现有值时才更新
                        if pd.notna(new_value) and (pd.isna(existing_value)
                                                    or float(new_value) > float(existing_value)):
                            # 更新所有以method开头的列
                            method_cols = [col for col in new_row.index if col.startswith(method)]
                            for col in method_cols:
                                merged_df.loc[merged_df['Dataset_id'] == dataset_id, col] = new_row[col]
                        elif pd.notna(new_value) and pd.notna(existing_value):
                            # 打印调试信息
                            print(f"Skipping update for {dataset_id}, {method}: "
                                  f"existing value ({existing_value}) >= new value ({new_value})")
            else:
                # 如果是新的Dataset_id，直接添加整行
                merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)

        merged_df.to_csv(output_file, index=False)
    else:
        new_df_processed.to_csv(output_file, index=False)


wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
if __name__ == "__main__":
    # Initialize wandb and set global configuration
    # Load dataset configuration and process results for tissue
    all_datasets = pd.read_csv(METADIR / "scdeepsort.csv", header=0, skiprows=[i for i in range(1, 68)])
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", type=str, default="Lung")
    args = parser.parse_args()
    tissue = args.tissue.capitalize()
    new_df = get_new_ans(tissue)
    write_ans(tissue, new_df)
