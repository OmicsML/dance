import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from dance.utils import try_import

wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
collect_datasets = {
    "cta_celltypist": [
        "c7775e88-49bf-4ba2-a03b-93f00447c958", "456e8b9b-f872-488b-871d-94534090a865",
        "738942eb-ac72-44ff-a64b-8943b5ecd8d9", "a5d95a42-0137-496f-8a60-101e17f263c8",
        "71be997d-ff75-41b9-8a9f-1288c865f921"
    ]
}
file_root = "/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning"


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


def write_ans():
    ans = []
    for method_folder in tqdm(collect_datasets):
        for dataset_id in collect_datasets[method_folder]:
            file_path = f"{file_root}/{method_folder}/{dataset_id}/results"
            step2_url = get_sweep_url(pd.read_csv(f"{file_path}/pipeline/best_test_acc.csv"))
            step3_urls = []
            for i in range(3):
                file_csv = f"{file_path}/params/{i}_best_test_acc.csv"
                if not os.path.exists(file_csv):  #no parameter
                    print(f"文件 {file_csv} 不存在，跳过。")
                    continue
                step3_urls.append(get_sweep_url(pd.read_csv(file_csv)))
            step3_str = ",".join(step3_urls)
            step_str = f"step2:{step2_url}|step3:{step3_str}"
            ans.append({"Dataset_id": dataset_id, method_folder: step_str})
    with open('temp_ans.json', 'w') as f:
        json.dump(ans, f, indent=4)


write_ans()
