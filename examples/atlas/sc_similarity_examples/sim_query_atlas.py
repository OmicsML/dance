import argparse
import os
import re
from pathlib import Path

import pandas as pd
import yaml

atlas_datasets = [
    "01209dce-3575-4bed-b1df-129f57fbc031", "055ca631-6ffb-40de-815e-b931e10718c0",
    "2a498ace-872a-4935-984b-1afa70fd9886", "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf",
    "3faad104-2ab8-4434-816d-474d8d2641db", "471647b3-04fe-4c76-8372-3264feb950e8",
    "4c4cd77c-8fee-4836-9145-16562a8782fe", "84230ea4-998d-4aa8-8456-81dd54ce23af",
    "8a554710-08bc-4005-87cd-da9675bdc2e7", "ae29ebd0-1973-40a4-a6af-d15a5f77a80f",
    "bc260987-8ee5-4b6e-8773-72805166b3f7", "bc2a7b3d-f04e-477e-96c9-9d5367d5425c",
    "d3566d6a-a455-4a15-980f-45eb29114cab", "d9b4bc69-ed90-4f5f-99b2-61b0681ba436",
    "eeacb0c1-2217-4cf6-b8ce-1f0fedf1b569"
]
import sys

sys.path.append("..")
import ast

from get_result_web import get_sweep_url, spilt_web

from dance import logger
from dance.utils import try_import

file_root = str(Path(__file__).resolve().parent.parent)


def find_unique_matching_row(df, config_col, input_dict_list):
    """Find a unique matching row in DataFrame based on specified criteria.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to search
    config_col : str
        Name of the DataFrame column containing dictionary list strings
    input_dict_list : list of dict
        Input dictionary list for matching

    Returns
    -------
    pandas.Series
        The matching row from the DataFrame

    Raises
    ------
    ValueError
        If the number of matching rows is not exactly one

    """

    # Define a function for parsing strings and comparing
    def is_match(config_str):
        try:
            # Safely parse string to Python object using ast.literal_eval
            config = ast.literal_eval(config_str)
            return config == input_dict_list
        except (ValueError, SyntaxError):
            # If parsing fails, no match
            return False

    # Apply comparison function to get a boolean series
    matches = df[config_col].apply(is_match)

    # Get all matching rows
    matching_rows = df[matches]

    # Check number of matching rows
    num_matches = len(matching_rows)
    if num_matches == 1:
        return matching_rows.iloc[0]
    elif num_matches == 0:
        raise ValueError("No matching rows found.")
    else:
        raise ValueError(f"Found {num_matches} matching rows, expected exactly one.")


wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
query_datasets = [
    "c7775e88-49bf-4ba2-a03b-93f00447c958", "456e8b9b-f872-488b-871d-94534090a865",
    "738942eb-ac72-44ff-a64b-8943b5ecd8d9", "a5d95a42-0137-496f-8a60-101e17f263c8",
    "71be997d-ff75-41b9-8a9f-1288c865f921"
]


def is_matching_dict(yaml_str, target_dict):

    # 解析YAML字符串
    yaml_config = yaml.safe_load(yaml_str)

    # 构建期望的字典格式
    expected_dict = {}
    for i, item in enumerate(yaml_config):
        if item['type'] == 'misc':  # 跳过misc类型
            continue
        key = f"pipeline.{i}.{item['type']}"
        value = item['target']
        expected_dict[key] = value

    # 直接比较两个字典是否相等
    return expected_dict == target_dict


def get_ans(query_dataset, method):
    result_path = f"{file_root}/tuning/{method}/{query_dataset}/results/atlas/best_test_acc.csv"
    if not os.path.exists(result_path):
        logger.warning(f"{result_path} not exists")
        return None
    data = pd.read_csv(result_path)
    sweep_url = get_sweep_url(data)
    _, _, sweep_id = spilt_web(sweep_url)
    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
    ans = pd.DataFrame(index=[method], columns=atlas_datasets)
    for i, run_kwarg in enumerate(sweep.config["parameters"]["run_kwargs"]["values"]):
        ans.loc[method, atlas_datasets[i]] = find_unique_matching_row(data, "run_kwargs", run_kwarg)["test_acc"]
        # ans.append({atlas_datasets[i]:find_unique_matching_row(data,"run_kwargs",run_kwarg)["test_acc"]})
    return ans


def get_ans_from_cache(query_dataset, method):
    #1:get best method from step2 of atlas datasets
    #2:search acc according to best method(需要注意的是，应该都是有值的，没有值的需要检查一下)
    ans = pd.DataFrame(index=method, columns=[f"{atlas_dataset}_from_cache" for atlas_dataset in atlas_datasets])
    sweep_url = re.search(r"step2:([^|]+)", conf_data[conf_data["Dataset_id"] == query_dataset][method]).group(1)
    _, _, sweep_id = spilt_web(sweep_url)
    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
    for atlas_dataset in atlas_datasets:
        best_yaml = conf_data[conf_data["Dataset_id"] == atlas_dataset][f"{method}_method"]
        match_run = None
        for run in sweep.runs:
            if is_matching_dict(best_yaml, run.config):
                if match_run is not None:
                    raise ValueError("match_run只能被赋值一次")
                else:
                    match_run = run
            if match_run is None:
                raise ValueError("未找到匹配")
        ans.loc[method, atlas_dataset] = match_run.summary["test_acc"]
    return ans


ans_all = {}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--methods", default=["cta_actinn", "cta_scdeepsort", "cta_singlecellnet", "cta_celltypist"],
                    nargs="+")
parser.add_argument("--tissue", type=str)
args = parser.parse_args()
methods = args.methods
tissue = args.tissue
conf_data = pd.read_excel("Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
if __name__ == "__main__":
    for query_dataset in query_datasets:
        ans = []
        for method in methods:
            ans.append(get_ans(query_dataset, method))
        ans = pd.concat(ans)
        ans_all[query_dataset] = ans
    for k, v in ans_all.items():
        v.to_csv(f"{str(methods)}_{k}_in_atlas.csv")
