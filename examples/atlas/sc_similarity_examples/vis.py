import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dance.utils import set_seed, try_import

sys.path.append("..")
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from get_result_web import spilt_web

wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tissue", type=str, default="blood")
args = parser.parse_args()
tissue = args.tissue
set_seed(42)
conf_data = pd.read_csv(f"results/{tissue}_result.csv", index_col=0)
query_datasets = list(conf_data[conf_data["queryed"] == True]["dataset_id"])

file_root = Path(__file__).resolve().parent
methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
feature_name = "spectral"
"""Visualization script for comparing model performance across different datasets and
methods.

This script loads experiment results from wandb and compares them with atlas-based
predictions, generating violin plots to visualize the distribution of accuracies.

"""


def get_accs(sweep):
    """Extract test accuracies from a wandb sweep.

    Parameters
    ----------
    sweep : wandb.Sweep
        Sweep object containing multiple runs

    Returns
    -------
    list
        List of test accuracies from all runs

    """
    ans = []
    for run in sweep.runs:
        if "test_acc" in run.summary:
            ans.append(run.summary["test_acc"])
    return ans


def get_runs(sweep_record):
    """Parse sweep URLs and collect all run results.

    Parameters
    ----------
    sweep_record : str
        String containing sweep URLs for different steps

    Returns
    -------
    list
        Combined list of test accuracies from all sweeps

    """
    step_links = {}
    pattern = r'(step\d+):((?:https?://[^|,]+(?:,)?)+)'
    matches = re.finditer(pattern, sweep_record)
    for match in matches:
        step = match.group(1)  # e.g., 'step2'
        links_str = match.group(2)  # e.g., 'https://...y31tzbnv'
        links = links_str.split(',')
        step_links[step] = links
    ans = []
    for step, links in step_links.items():
        for sweep_url in links:
            _, _, sweep_id = spilt_web(sweep_url)
        sweep = wandb.Api(timeout=1000).sweep(f"{entity}/{project}/{sweep_id}")
        ans += get_accs(sweep)
    return ans


def get_atlas_ans(query_dataset, method):
    """Calculate atlas-based prediction accuracy for a given dataset and method.

    Parameters
    ----------
    query_dataset : str
        Dataset identifier
    method : str
        Method name to evaluate

    Returns
    -------
    float
        Predicted accuracy based on atlas similarity

    """
    data = pd.read_excel(f"{tissue}_similarity.xlsx", sheet_name=query_dataset[:4], index_col=0)
    weight1 = 1.0  # Weight for feature-based similarity
    weight2 = 0.0  # Weight for metadata similarity
    weighted_sum = data.loc[feature_name, :] * weight1 + data.loc["metadata_sim", :] * weight2
    atlas_dataset_res = weighted_sum.idxmax()  # Get most similar dataset
    max_value = weighted_sum.max()
    if method in data.index:
        return data.loc[method, atlas_dataset_res]
    else:
        return 0


def vis(data, target_value, title, ax):
    """Create violin plot comparing distribution of accuracies with atlas prediction.

    Parameters
    ----------
    data : list
        List of accuracy values
    target_value : float
        Atlas-predicted accuracy value
    title : str
        Plot title
    ax : matplotlib.axes.Axes
        Axes object to plot on

    """
    # sns.boxplot(data=data, color='skyblue',ax=ax)
    # if target_value is not np.nan:
    #     ax.axhline(y=target_value, color='red', linestyle='--', linewidth=2, label=f'atlas_value = {target_value}')
    #     ax.text(0, target_value + (max(data)-min(data))*0.01, f'{target_value}', color='red', ha='center',size=16)

    data = np.array(data)
    data_df = pd.DataFrame({'test_acc': data})
    sns.violinplot(y='test_acc', data=data_df, inner=None, color='skyblue', ax=ax)
    median = np.median(data)
    ax.axhline(median, color='gray', linestyle='--', label=f'Median: {median:.1f}')
    if not np.isnan(target_value):
        percentile = (np.sum(data < float(target_value)) / len(data)) * 100
        ax.scatter(0, float(target_value), color='red', s=100, zorder=5,
                   label=f'Specific Value: {target_value}\n({percentile:.1f} percentile)')
    ax.set_title(str(title))
    ax.set_ylabel('test_acc')
    ax.title.set_size(16)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend()


def get_runs(query_dataset, method):
    step_str = conf_data[conf_data["dataset_id"] == query_dataset][method].iloc[0]
    step2_str = step_str.split("step2:")[1].split("|")[0]
    _, _, sweep_id = spilt_web(step2_str)
    sweep = wandb.Api(timeout=1000).sweep(f"{entity}/{project}/{sweep_id}")
    runs = []
    for run in sweep.runs:
        if "test_acc" in run.summary:
            runs.append(run.summary["test_acc"])
    return runs


if __name__ == "__main__":
    # ans_all=defaultdict(dict)
    # for query_dataset in query_datasets:
    #     for method in methods:
    #         sweep_record=ground_truth_conf.loc[query_dataset,method]
    #         ans_all[query_dataset][method]=get_runs(sweep_record)
    # with open("runs.json","w") as f:
    #     json.dump(ans_all,f)

    # with open("runs.json") as f:
    #     runs = json.load(f)
    plt.style.use("default")

    # Generate visualization for each dataset
    for query_dataset in query_datasets:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        # Create subplot for each method
        for i, method in enumerate(methods):
            vis(get_runs(query_dataset, method), get_atlas_ans(query_dataset, method), f"{query_dataset}_{method}",
                axes[i])
        plt.tight_layout()
        plt.savefig(f"imgs/{query_dataset}.png", dpi=300)
        plt.show()
