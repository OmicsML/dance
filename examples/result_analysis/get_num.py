import sys
from pathlib import Path

import pandas as pd

sys.path.append("..")
import urllib

from get_result_web import spilt_web

from dance.utils import try_import

wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
tasks = ["cell type annotation new", "clustering", "imputation_new", "spatial domain", "cell type deconvolution"]
file_root = Path(__file__).resolve().parent
prefix = 'https://wandb.ai/xzy11632/dance-dev'

runs_sum = 0

for task in tasks:
    data = pd.read_excel(file_root / "results.xlsx", sheet_name=task, dtype=str)
    matched_list = data.applymap(lambda x: x if isinstance(x, str) and x.startswith(prefix) else None).stack().tolist()
    for sweep_url in matched_list:
        _, _, sweep_id = spilt_web(sweep_url)
        print(sweep_id)
        sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
        runs_sum += (len(sweep.runs))
print(runs_sum)
