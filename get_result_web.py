import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from dance.utils import try_import

# os.environ["http_proxy"]="http://121.250.209.147:7890"
# os.environ["https_proxy"]="http://121.250.209.147:7890"
wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
collect_datasets = {
    "cta_actinn": [
        "84230ea4-998d-4aa8-8456-81dd54ce23af", "d3566d6a-a455-4a15-980f-45eb29114cab",
        "4c4cd77c-8fee-4836-9145-16562a8782fe", "ae29ebd0-1973-40a4-a6af-d15a5f77a80f",
        "bc260987-8ee5-4b6e-8773-72805166b3f7", "bc2a7b3d-f04e-477e-96c9-9d5367d5425c",
        "d9b4bc69-ed90-4f5f-99b2-61b0681ba436"
    ],
    "cta_celltypist": [
        "4c4cd77c-8fee-4836-9145-16562a8782fe", "ae29ebd0-1973-40a4-a6af-d15a5f77a80f",
        "bc260987-8ee5-4b6e-8773-72805166b3f7", "bc2a7b3d-f04e-477e-96c9-9d5367d5425c",
        "d9b4bc69-ed90-4f5f-99b2-61b0681ba436", "01209dce-3575-4bed-b1df-129f57fbc031",
        "055ca631-6ffb-40de-815e-b931e10718c0", "2a498ace-872a-4935-984b-1afa70fd9886",
        "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf", "3faad104-2ab8-4434-816d-474d8d2641db"
    ],
    "cta_singlecellnet": [
        "4c4cd77c-8fee-4836-9145-16562a8782fe", "ae29ebd0-1973-40a4-a6af-d15a5f77a80f",
        "bc260987-8ee5-4b6e-8773-72805166b3f7", "bc2a7b3d-f04e-477e-96c9-9d5367d5425c",
        "d9b4bc69-ed90-4f5f-99b2-61b0681ba436", "01209dce-3575-4bed-b1df-129f57fbc031",
        "055ca631-6ffb-40de-815e-b931e10718c0", "2a498ace-872a-4935-984b-1afa70fd9886",
        "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf", "3faad104-2ab8-4434-816d-474d8d2641db"
    ]
}
file_root = "/home/zyxing/dance/examples/tuning"


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
def get_sweep_url(step_csv: pd.DataFrame):
    ids = step_csv["id"]
    sweep_urls = []
    for run_id in tqdm(ids, leave=False):
        api = wandb.Api()
        run = api.run(f"/{entity}/{project}/runs/{run_id}")
        sweep_urls.append(run.sweep.url)
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
                step3_urls.append(get_sweep_url(pd.read_csv(f"{file_path}/params/{i}_best_test_acc.csv")))
            step3_str = ",".join(step3_urls)
            step_str = f"step2:{step2_url}|step3:{step3_str}"
            ans.append({"Dataset_id": dataset_id, method_folder: step_str})
    with open('temp_ans.json', 'w') as f:
        json.dump(ans, f)


write_ans()
