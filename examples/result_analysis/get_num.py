import os
import pickle

import pandas as pd

from dance import logger
from dance.pipeline import get_additional_sweep, save_summary_data
from dance.settings import EXAMPLESDIR
from dance.utils import spilt_web

metrics_dict = [{
    "task": "celltype annotation",
    "metric": "test_acc",
    "ascending": False
}, {
    "task": "cluster",
    "metric": "acc",
    "ascending": False
}, {
    "task": "imputation",
    "metric": "test_MRE",
    "ascending": True
}, {
    "task": "spatial domain",
    "metric": "ARI",
    "ascending": False
}, {
    "task": "celltype deconvolution",
    "metric": "test_MSE",
    "ascending": True
}, {
    "task": "joint embedding",
    "metric": "ARI",
    "ascending": False
}]
tasks = [d["task"] for d in metrics_dict]
choose_tasks = tasks

mertic_names = [d["metric"] for d in metrics_dict]
ascendings = [d["ascending"] for d in metrics_dict]
entity = "xzy11632"
project = "dance-dev"
prefix = f'https://wandb.ai/{entity}/{project}'
ans_all = {}
error_all = {}
run_counts = 0
step2_run_task_counts = {}
if os.path.exists(EXAMPLESDIR / 'result_analysis/migration/cache/sweep_cache_data.pkl'):
    with open(EXAMPLESDIR / 'result_analysis/migration/cache/sweep_cache_data.pkl', 'rb') as file:
        sweep_cache_data = pickle.load(file)
else:
    sweep_cache_data = {}
for i, task in enumerate(tasks):
    # Skip tasks not in choose_tasks list
    if task not in choose_tasks:
        continue
    # Read and preprocess results from Excel file
    data = pd.read_excel(EXAMPLESDIR / "result_analysis/results.xlsx", sheet_name=task, dtype=str)
    data['Methods'] = data['Methods'].fillna(method='ffill')
    data['step name'] = data['step name'].fillna(method='ffill')
    data = data.set_index(['Methods'])
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):

            # Extract metadata
            method = data.index[row_idx]
            dataset = data.columns[col_idx]
            value = data.iloc[row_idx, col_idx]
            step_name = data.iloc[row_idx]["step name"]
            if isinstance(value, str) and value.startswith(prefix) and (str(step_name).lower() == "step 2"
                                                                        or str(step_name).lower() == "step 3"):
                sweep_url = value
                logger.info(f"sweep_url:{sweep_url}")
            else:
                continue
            _, _, sweep_id = spilt_web(sweep_url)
            summary_data = []
            if sweep_id in sweep_cache_data:
                summary_data = sweep_cache_data[sweep_id]
            elif str(step_name).lower() == "step 2":
                sweep_ids = get_additional_sweep(entity=entity, project=project, sweep_id=sweep_id)
                try:
                    sweep_ids.remove(sweep_id)
                    summary_data = save_summary_data(entity, project, sweep_id=sweep_id, summary_file_path="",
                                                     root_path="", save=False, additional_sweep_ids=sweep_ids)
                    sweep_cache_data[sweep_id] = summary_data
                except Exception as e:
                    print(e)
                    error_all[(task, method, dataset, sweep_id)] = str(e)
            else:
                try:
                    summary_data = save_summary_data(entity, project, sweep_id=sweep_id, summary_file_path="",
                                                     root_path="", save=False)
                    sweep_cache_data[sweep_id] = summary_data
                except Exception as e:
                    print(e)
                    error_all[(task, method, dataset, sweep_id)] = str(e)
            if str(step_name).lower() == "step 2":
                if task not in step2_run_task_counts:
                    step2_run_task_counts[task] = len(summary_data)
                else:
                    step2_run_task_counts[task] += len(summary_data)
            run_counts += len(summary_data)
            if (task, method, dataset) not in ans_all:
                ans_all[(task, method, dataset)] = [summary_data]
            else:
                ans_all[(task, method, dataset)].append(summary_data)
print(f"Step 2 Run task counts: {step2_run_task_counts}")
print(f"Step 2 Sum task counts: {sum(step2_run_task_counts.values())}")
print(f"Total runs processed: {run_counts}")
with open(EXAMPLESDIR / 'result_analysis/migration/cache/sweep_cache_data.pkl', 'wb') as f:
    pickle.dump(sweep_cache_data, f)
