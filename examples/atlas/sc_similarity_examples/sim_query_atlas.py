import argparse
from pathlib import Path

import pandas as pd

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

from dance.utils import try_import

file_root = str(Path(__file__).resolve().parent.parent)


def find_unique_matching_row(df, config_col, input_dict_list):
    """在 DataFrame 中查找指定列中与输入字典列表匹配的唯一一行。

    :param df: pandas.DataFrame，包含要搜索的数据。
    :param config_col: str，DataFrame 中包含字典列表字符串的列名。
    :param input_dict_list: list of dicts，输入的字典列表，用于匹配。
    :return: pandas.Series，匹配的行。
    :raises ValueError: 如果匹配的行数不等于1。

    """

    # 定义一个函数，用于解析字符串并比较
    def is_match(config_str):
        try:
            # 使用 ast.literal_eval 安全地解析字符串为 Python 对象
            config = ast.literal_eval(config_str)
            return config == input_dict_list
        except (ValueError, SyntaxError):
            # 如果解析失败，则不匹配
            return False

    # 应用比较函数，得到一个布尔系列
    matches = df[config_col].apply(is_match)

    # 获取所有匹配的行
    matching_rows = df[matches]

    # 检查匹配的行数
    num_matches = len(matching_rows)
    if num_matches == 1:
        return matching_rows.iloc[0]
    elif num_matches == 0:
        raise ValueError("未找到匹配的行。")
    else:
        raise ValueError(f"找到 {num_matches} 行匹配，预期恰好一行。")


wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
query_datasets = [
    "c7775e88-49bf-4ba2-a03b-93f00447c958", "456e8b9b-f872-488b-871d-94534090a865",
    "738942eb-ac72-44ff-a64b-8943b5ecd8d9", "a5d95a42-0137-496f-8a60-101e17f263c8",
    "71be997d-ff75-41b9-8a9f-1288c865f921"
]


def get_ans(query_dataset, method):
    data = pd.read_csv(f"{file_root}/tuning/{method}/{query_dataset}/results/atlas/best_test_acc.csv")
    sweep_url = get_sweep_url(data)
    _, _, sweep_id = spilt_web(sweep_url)
    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
    ans = pd.DataFrame(index=[method], columns=atlas_datasets)
    for i, run_kwarg in enumerate(sweep.config["parameters"]["run_kwargs"]["values"]):
        ans.loc[method, atlas_datasets[i]] = find_unique_matching_row(data, "run_kwargs", run_kwarg)["test_acc"]
        # ans.append({atlas_datasets[i]:find_unique_matching_row(data,"run_kwargs",run_kwarg)["test_acc"]})
    return ans


ans_all = {}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--methods", default=["cta_actinn", "cta_scdeepsort"], nargs="+")
args = parser.parse_args()
methods = args.methods
if __name__ == "__main__":
    for query_dataset in query_datasets:
        ans = []
        for method in methods:
            ans.append(get_ans(query_dataset, method))
        ans = pd.concat(ans)
        ans_all[query_dataset] = ans
    for k, v in ans_all.items():
        v.to_csv(f"{str(methods)}_{k}_in_atlas.csv")
