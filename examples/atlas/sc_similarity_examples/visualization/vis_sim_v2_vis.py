import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dance.settings import ATLASDIR, SIMILARITYDIR

sys.path.append(str(SIMILARITYDIR))
sys.path.append(str(ATLASDIR))
from similarity.process_tissue_similarity_matrices import convert_to_complex
from visualization.vis_sim_v2_data import exclude_data, get_ans, get_atlas_acc, get_atlas_ans

similarity_names = {
    "wasserstein": "Wasserstein similarity",
    "Hausdorff": "Hausdorff similarity",
    "chamfer": "Chamfer similarity",
    "energy": "Energy similarity",
    "sinkhorn2": "Sinkhorn similarity",
    "bures": "Bures similarity",
    "spectral": "Spectral similarity",
    "mmd": "MMD similarity"
}


def plot_pre_normalized_radar_v3(
    df,
    highlight_dataset_name,
    tissue,
    query_dataset,
    # Control area
    highlight_fill=False,
    other_fill=False,
    # Colors and transparency
    highlight_color='crimson',
    other_color='skyblue',
    other_alpha=0.20,
    highlight_fill_alpha=0.45,
    # Line width
    highlight_linewidth=0.5,
    other_linewidth=0.5,
    # Size and font
    figsize=(15, 10),  # Suggest slightly increasing figure size to accommodate large fonts
    title="Performance Radar",
    title_fontsize=18,
    label_fontsize=17,  # Now can safely increase font size
    tick_label_fontsize=16,
    # Y-axis control
    ylim_min=0.0,
    ylim_max=1.0,
    num_yticks=4,
    # ==================== New core parameters ====================
    label_distance_factor=1.1  # Control distance between labels and chart edge
    # (1.0 = at edge, 1.2 = 20% radius from edge)
    # ====================================================
):
    """Draw radar chart (version 6).

    New features:
    - Smart label layout: Manually place axis labels to avoid overlap with chart, allowing larger fonts.
    - label_distance_factor: Control distance between labels and chart edge.

    """

    # --- 1. Input validation ---
    if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
    if df.empty: raise ValueError("Input DataFrame is empty.")
    if highlight_dataset_name not in df.index: raise ValueError(f"Dataset '{highlight_dataset_name}' not found.")

    # --- 2. Prepare data ---
    features_raw = df.columns.tolist()
    features = [similarity_names.get(f, f) for f in features_raw]
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # --- Auto-calculate Y-axis lower limit ---
    final_ylim_min = ylim_min
    if final_ylim_min is None:
        global_min = df.min().min()
        padding = (df.max().max() - global_min) * 0.1
        final_ylim_min = max(0, global_min - padding)
        print(f"Auto-calculated ylim_min: {final_ylim_min:.3f}")

    # --- 3. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Plot other and highlighted datasets
    for index, row in df.iterrows():
        is_highlight = (index == highlight_dataset_name)
        values = row.tolist()
        values += values[:1]

        ax.plot(angles, values, color=highlight_color if is_highlight else other_color,
                linewidth=highlight_linewidth if is_highlight else other_linewidth, zorder=4 if is_highlight else 2,
                label=f"{index}" if is_highlight else None)

        if is_highlight and highlight_fill:
            ax.fill(angles, values, color=highlight_color, alpha=highlight_fill_alpha, zorder=3)
        elif not is_highlight and other_fill:
            ax.fill(angles, values, color=other_color, alpha=other_alpha, zorder=1)

    # --- 4. Set axes and labels (core modification part) ---

    # Set Y-axis range and ticks
    ax.set_ylim(final_ylim_min, ylim_max)
    yticks = np.linspace(final_ylim_min, ylim_max, num_yticks)
    yticklabels = [f"{tick:.2f}" for tick in yticks]
    if ylim_max == 1.0: yticklabels[-1] = f"{yticklabels[-1]} (Best)"
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=tick_label_fontsize, color="black")
    ax.set_rlabel_position(90)  # Place Y-axis labels on the right

    # Set X-axis (angle axis) grid lines, but hide default labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # <--- Key: disable default labels!

    # ==================== Manual placement and alignment of labels ====================
    for i, angle in enumerate(angles[:-1]):
        angle_deg = np.rad2deg(angle)

        # Determine horizontal alignment
        if angle_deg == 0: ha = 'left'
        elif angle_deg == 90: ha = 'right'
        elif 0 < angle_deg < 90: ha = 'left'
        elif 270 < angle_deg < 360: ha = 'left'
        else: ha = 'right'

        # Determine vertical alignment
        if angle_deg in [0, 180]: va = 'center'
        elif 90 < angle_deg < 270: va = 'top'
        else: va = 'bottom'

        ax.text(
            angle,
            ylim_max * label_distance_factor,  # Place outside the chart
            features[i],
            size=label_fontsize,
            ha=ha,
            va=va,
            fontweight='bold')  # Bold to make it clearer
    # ==========================================================

    # --- 5. Add legend and title ---
    # Adjust position to accommodate larger fonts and farther labels
    ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1.2), fontsize=label_fontsize)
    # plt.title(title, size=title_fontsize, y=1.2, weight='bold')

    # --- 6. Save and display ---
    # `tight_layout` may not work well after manual layout, can skip or call at the end
    fig.tight_layout()
    if query_dataset is not None:
        img_path = SIMILARITYDIR / f"data/imgs/sim_imgs/{tissue}/{query_dataset}_radar_plot.pdf"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300, format="pdf")
        print(f"Radar plot saved to: {img_path}")
    plt.show()

    return fig, ax


if __name__ == "__main__":
    tissues = ["blood", "brain", "heart", "intestine", "kidney", "lung", "pancreas"]
    for tissue in tissues:
        conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
        query_datasets = list(conf_data[conf_data["queryed"] == True]["dataset_id"])
        for exclude_dataset in exclude_data.get(tissue, []):
            if exclude_dataset in query_datasets:
                query_datasets.remove(exclude_dataset)
        ans = get_ans(query_datasets, tissue, exclude_data=exclude_data)
        methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
        get_atlas_acc(ans, methods)
        feature_name = "wasserstein"
        feature_names = ["wasserstein", "Hausdorff", "chamfer", "energy", "sinkhorn2", "bures", "spectral", "mmd"]
        # feature_names.append("metadata_sim")
        # feature_names.append("average_acc")
        for query_dataset, data in ans.items():
            methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
            for i, method_key in enumerate(methods):  # Renamed 'method' to 'method_key'
                target_value_str, current_atlas_dataset = get_atlas_ans(query_dataset, method_key,
                                                                        feature_name=feature_name,
                                                                        data=data)  # only for current_atlas_dataset
            df_sim = data.loc[feature_names, :].T.applymap(convert_to_complex)
            print(query_dataset)
            plot_pre_normalized_radar_v3(df_sim, current_atlas_dataset, tissue=tissue, query_dataset=query_dataset,
                                         title_fontsize=14, other_fill=False)
            break
        break
