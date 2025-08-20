import argparse
import os
import re
import sys
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.api.types as ptypes  # Import Pandas type API
import scipy.sparse as sp

from dance.atlas.sc_similarity.anndata_similarity import get_anndata
from dance.pipeline import get_additional_sweep
from dance.settings import ATLASDIR, EXAMPLESDIR, SIMILARITYDIR
from dance.utils import set_seed, spilt_web, try_import

sys.path.append(str(SIMILARITYDIR))
sys.path.append(str(ATLASDIR))
import json

import matplotlib.pyplot as plt
import seaborn as sns
from similarity.process_tissue_similarity_matrices import convert_to_complex

wandb = try_import("wandb")

# tissue_count= 0
from dance.settings import entity, project


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
    reduce_error = False
    in_query = False
    with open(
            SIMILARITYDIR /
            f"data/similarity_weights_results/{'reduce_error_' if reduce_error else ''}{'in_query_' if in_query else ''}sim_dict.json",
            encoding='utf-8') as f:
        sim_dict = json.load(f)
    feature_name = sim_dict[tissue]["feature_name"]
    data = pd.read_excel(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx", sheet_name=query_dataset[:4],
                         index_col=0)
    weight1 = sim_dict[tissue]["weight1"]  # Weight for feature-based similarity
    weight2 = 1 - weight1  # Weight for metadata similarity
    data.loc[feature_name, :] = data.loc[feature_name, :].apply(convert_to_complex)
    weighted_sum = (data.loc[feature_name, :] * weight1 + data.loc["metadata_sim", :] * weight2).astype(float)
    atlas_dataset_res = weighted_sum.idxmax()  # Get most similar dataset
    max_value = weighted_sum.max()
    if method in data.index:
        return data.loc[method, atlas_dataset_res], atlas_dataset_res
    else:
        return 0, "null"


def plot_pre_normalized_radar_v3(
        df,
        highlight_dataset_name,
        # Control highlight area
        highlight_fill=False,
        # New parameter: control other areas
        other_fill=True,
        highlight_color='crimson',
        other_color='skyblue',
        other_alpha=0.20,
        highlight_fill_alpha=0.45,
        highlight_linewidth=2.5,
        other_linewidth=1.0,
        figsize=(10, 10),
        title="Performance Radar",
        title_fontsize=16,
        label_fontsize=10,
        tick_label_fontsize=9):
    """Draw radar chart, directly using pre-normalized data (version 3).

    New features:
    - highlight_fill (bool): Control whether to fill the highlight area.
    - other_fill (bool): Control whether to fill other non-highlight areas.

    """

    # --- 1. Input validation ---
    # (Omitted validation code same as v2...)
    if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
    if df.empty: raise ValueError("Input DataFrame is empty.")
    if highlight_dataset_name not in df.index: raise ValueError(f"Dataset '{highlight_dataset_name}' not found.")
    if not ((df.min().min() >= 0) and (df.max().max() <= 1)): raise ValueError("All values must be between 0 and 1.")

    # --- 2. Prepare plotting ---
    features = df.columns.tolist()
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # --- 3. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Plot other datasets
    for index, row in df.iterrows():
        if index == highlight_dataset_name:
            continue
        values = row.tolist()
        values += values[:1]

        # Plot contour lines of other datasets
        ax.plot(angles, values, color=other_color, linewidth=other_linewidth, zorder=2)

        # ==================== Core modifications for other areas ====================
        # Decide whether to fill based on other_fill parameter
        if other_fill:
            ax.fill(angles, values, color=other_color, alpha=other_alpha, zorder=1)  # Lower the zorder of fill

    # Plot highlighted dataset (always on top layer)
    highlight_values = df.loc[highlight_dataset_name].tolist()
    highlight_values += highlight_values[:1]
    ax.plot(angles, highlight_values, color=highlight_color, linewidth=highlight_linewidth, zorder=4,
            label=f"{highlight_dataset_name} (Highlighted)")

    if highlight_fill:
        ax.fill(angles, highlight_values, color=highlight_color, alpha=highlight_fill_alpha, zorder=3)

    # --- 4. Set axes and labels ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=label_fontsize)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0 (Best)"], fontsize=tick_label_fontsize, color="grey")
    ax.set_rlabel_position(180 / num_vars if num_vars > 1 else 45)

    # --- 5. Add legend and title ---
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15))
    plt.title(title, size=title_fontsize, y=1.15, weight='bold')
    plt.tight_layout(pad=2.0)
    plt.show()

    return fig, ax


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--tissue", type=str, default="blood")
    # args = parser.parse_args()
    # tissue = args.tissue
    tissue = "pancreas"
    set_seed(42)
    conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
    query_datasets = list(conf_data[conf_data["queryed"] == True]["dataset_id"])

    plt.style.use("default")
    import json
    ans = {}
    feature_names = ["wasserstein", "Hausdorff", "chamfer", "energy", "sinkhorn2", "bures", "spectral", "mmd"]
    feature_names.append("metadata_sim")
    for query_dataset in query_datasets:
        data = pd.read_excel(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx", sheet_name=query_dataset[:4],
                             index_col=0)
        ans[query_dataset] = data
        methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
        for i, method_key in enumerate(methods):  # Renamed 'method' to 'method_key'
            target_value_str, current_atlas_dataset = get_atlas_ans(query_dataset,
                                                                    method_key)  # only for current_atlas_dataset
        df_sim = data.loc[feature_names, :].T.applymap(convert_to_complex)
        # try:
        plot_pre_normalized_radar_v3(df_sim, current_atlas_dataset, title_fontsize=14, other_fill=False)
        # except Exception as e:
        #     print(f"Error in Scenario 2: {e}")
