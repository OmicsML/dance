import argparse
import os
import re
import sys
from math import e
from pathlib import Path

import numpy as np
import pandas as pd
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
from visualization.vis_sim_v2_data import exclude_data, get_ans, get_atlas_ans

wandb = try_import("wandb")

# tissue_count= 0
from dance.settings import entity, project


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


# def vis(data, target_value, title, ax):
#     """Create violin plot comparing distribution of accuracies with atlas prediction.

#     Parameters
#     ----------
#     data : list
#         List of accuracy values
#     target_value : float
#         Atlas-predicted accuracy value
#     title : str
#         Plot title
#     ax : matplotlib.axes.Axes
#         Axes object to plot on

#     """
#     # sns.boxplot(data=data, color='skyblue',ax=ax)
#     # if target_value is not np.nan:
#     #     ax.axhline(y=target_value, color='red', linestyle='--', linewidth=2, label=f'atlas_value = {target_value}')
#     #     ax.text(0, target_value + (max(data)-min(data))*0.01, f'{target_value}', color='red', ha='center',size=16)


#     data = np.array(data)
#     data_df = pd.DataFrame({'test_acc': data})
#     sns.violinplot(y='test_acc', data=data_df, inner=None, color='skyblue', ax=ax)
#     median = np.median(data)
#     ax.axhline(median, color='gray', linestyle='--', label=f'Median: {median:.4f}')
#     if np.isnan(target_value):
#         target_value = -0.01
#     percentile = (np.sum(data < float(target_value)) / len(data)) * 100
#     ax.scatter(0, float(target_value), color='red', s=100, zorder=5,
#                label=f'Specific Value: {target_value:.4f}\n({percentile:.1f} percentile)')
#     ax.set_title(str(title))
#     ax.set_ylabel('test_acc')
#     ax.title.set_size(16)
#     ax.yaxis.label.set_size(14)
#     ax.tick_params(axis='both', which='major', labelsize=10)
#     ax.legend()
def vis(data, target_value, title, ax):
    """Create box plot comparing distribution of accuracies with atlas prediction.

    Parameters
    ----------
    data : list or np.ndarray
        List of accuracy values
    target_value : float
        Atlas-predicted accuracy value. If np.nan, the line and its associated
        percentile information will not be shown.
    title : str
        Plot title
    ax : matplotlib.axes.Axes
        Axes object to plot on

    """
    # 将输入数据转换为 NumPy 数组，方便后续处理
    data_np = np.array(data)
    # 为 Seaborn 创建 DataFrame
    # 确保 'test_acc' 列是 float 类型，即使 data_np 为空也能正常处理
    data_df = pd.DataFrame({'test_acc': data_np.astype(float)})

    # 1. 将小提琴图改成箱线图
    sns.boxplot(y='test_acc', data=data_df, color='skyblue', ax=ax)

    # 2. target_value 改成线，并将百分位数放到标签里
    #    仅当 target_value 是一个有效数字时执行
    tv_float = float(target_value)  # 将 target_value 转换为 float 类型进行判断和计算

    if not np.isnan(tv_float):
        # 准备 target_value 线的标签文本
        label_for_target_line = f'Atlas Value: {tv_float:.4f}'
        percentile_info_str = ""

        # 仅当数据列表不为空时，计算百分位数
        if len(data_np) > 0:
            percentile = (np.sum(data_np <= tv_float) / len(data_np)) * 100
            percentile_info_str = f"\n({percentile:.1f}% percentile)"  # 使用换行符使图例更易读
            label_for_target_line += percentile_info_str

        # 绘制 target_value 的水平线
        ax.axhline(y=tv_float, color='red', linestyle='--', linewidth=2, label=label_for_target_line)

        # 在图上添加 target_value 的文本标注 (与图例中的标签分开)
        # 计算文本标注的 y 轴位置，使其略高于红线
        text_y_position = tv_float  # 默认位置
        if len(data_np) > 0:
            data_min, data_max = np.min(data_np), np.max(data_np)
            if (data_max - data_min) > 1e-9:  # 检查数据是否有跨度
                text_y_position = tv_float + (data_max - data_min) * 0.02
            else:  # 数据无跨度 (所有值相同) 或跨度极小
                text_y_position = tv_float + (abs(tv_float * 0.02) if abs(tv_float) > 1e-9 else 0.002)
        else:  # 数据为空
            text_y_position = tv_float + (abs(tv_float * 0.02) if abs(tv_float) > 1e-9 else 0.002)

        ax.text(0, text_y_position, f'{tv_float:.4f}', color='red', ha='center', va='bottom', size=12)

    # 3. 设置标题和标签 (保持不变)
    ax.set_title(str(title))
    ax.set_ylabel('test_acc')
    ax.title.set_size(16)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # 4. 显示图例 (保持不变)
    #    为了避免在没有可显示图例项时出现警告，可以先检查是否有句柄
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # 仅当有图例项时显示图例
        ax.legend()


def get_runs(conf_data, query_dataset, method):
    cache_file = SIMILARITYDIR / "cache/sweep_cache.json"
    step_str = conf_data[conf_data["dataset_id"] == query_dataset][method].iloc[0]
    if pd.isna(step_str):
        return None
    step2_str = step_str.split("step2:")[1].split("|")[0]
    _, _, sweep_id = spilt_web(step2_str)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            sweep_cache = json.load(f)
    else:
        sweep_cache = {}
    # print(sweep_id)
    sweep_ids = get_additional_sweep(entity=entity, project=project, sweep_id=sweep_id)
    runs = []

    for sweep_id in sweep_ids:
        if sweep_id in sweep_cache:
            runs.extend(sweep_cache[sweep_id])
        else:
            sweep = wandb.Api(timeout=1000).sweep(f"{entity}/{project}/{sweep_id}")
            sweep_runs = []
            for run in sweep.runs:
                if "test_acc" in run.summary:
                    sweep_runs.append(run.summary["test_acc"])
                else:
                    # sweep_runs.append(-0.01)
                    sweep_runs.append(0)

            sweep_cache[sweep_id] = sweep_runs
            with open(cache_file, 'w') as f:
                json.dump(sweep_cache, f)
            runs.extend(sweep_runs)
    return runs


def plot_umap(query_dataset, atlas_dataset, save_path):
    """Create UMAP plot comparing two datasets.

    Parameters
    ----------
    adata1 : anndata.AnnData
        First dataset to plot
    adata2 : anndata.AnnData
        Second dataset to plot

    """
    import matplotlib.pyplot as plt
    import umap
    print(query_dataset, atlas_dataset)
    query_data = get_anndata(train_dataset=[f"{query_dataset}"], data_dir=f"{EXAMPLESDIR}/tuning/temp_data",
                             tissue=tissue.capitalize())
    atlas_data = get_anndata(train_dataset=[f"{atlas_dataset}"], data_dir=f"{EXAMPLESDIR}/tuning/temp_data",
                             tissue=tissue.capitalize())
    reducer = umap.UMAP()
    if sp.issparse(query_data.X):
        query_data.X = query_data.X.toarray()
    if sp.issparse(atlas_data.X):
        atlas_data.X = atlas_data.X.toarray()
    combined_data = np.concatenate([query_data.X, atlas_data.X], axis=0)
    embedding = reducer.fit_transform(combined_data)

    plt.figure(figsize=(10, 5))
    plt.scatter(embedding[:query_data.shape[0], 0], embedding[:query_data.shape[0], 1], label=query_dataset, alpha=0.5)
    plt.scatter(embedding[query_data.shape[0]:, 0], embedding[query_data.shape[0]:, 1], label=atlas_dataset, alpha=0.5)
    plt.legend()
    plt.title('UMAP Projection of Two Datasets')
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_combined_methods(data, query_dataset, methods, tissue, feature_name, conf_data, save=True,
                          method_runs_cache=None, overall_data_tissue=None):
    fig, ax = plt.subplots(figsize=(4, 3))  # Slightly larger for clarity

    plot_data_list = []
    target_details = []
    method_names_for_plot = []
    # CRITICAL FIX: Initialize all_runs_data_for_methods *before* the loop
    all_runs_data_for_methods = {}
    atlas_dataset_for_label = "Unknown Atlas"  # To store a representative atlas name

    vis_dict = {
        "cta_actinn": "ACTINN",
        "cta_celltypist": "Celltypist",
        "cta_scdeepsort": "ScDeepsort",
        "cta_singlecellnet": "singleCellNet"
    }

    for i, method_key in enumerate(methods):  # Renamed 'method' to 'method_key'
        target_value_str, current_atlas_dataset = get_atlas_ans(query_dataset, method_key, feature_name, data)
        if i == 0 and current_atlas_dataset:  # Capture atlas name from the first method
            atlas_dataset_for_label = current_atlas_dataset
        if method_runs_cache is not None and method_key in method_runs_cache:
            runs = method_runs_cache[method_key]
        else:
            runs = get_runs(conf_data, query_dataset, method_key)
        current_method_label = vis_dict.get(method_key, method_key)  # Safer get

        if not runs:
            print(
                f"No runs data for {query_dataset} with method {method_key} ('{current_method_label}'). Skipping boxplot."
            )
            try:
                tv_float_check = float(target_value_str)
                if not np.isnan(tv_float_check):
                    print(f"  (Target value {tv_float_check:.4f} exists but no run data for {method_key})")
            except (ValueError, TypeError):
                # If target_value_str itself is not a valid float representation
                pass  # Silently ignore if target also unparsable and no runs
            continue

        method_names_for_plot.append(current_method_label)
        # Ensure runs are stored as a NumPy array for consistent calculations later
        all_runs_data_for_methods[current_method_label] = np.array(runs)

        for run_val in runs:
            plot_data_list.append({'method': current_method_label, 'accuracy': run_val})

        try:
            tv_float = float(target_value_str)
        except (ValueError, TypeError):
            print(
                f"Warning: Could not convert target_value '{target_value_str}' to float for method {method_key}. Skipping target details."
            )
            tv_float = np.nan  # Treat as NaN if unparsable

        if not np.isnan(tv_float):
            # runs_np is already stored in all_runs_data_for_methods[current_method_label]
            runs_for_percentile = all_runs_data_for_methods[current_method_label]
            percentile = (np.sum(runs_for_percentile <= tv_float) /
                          len(runs_for_percentile)) * 100 if len(runs_for_percentile) > 0 else np.nan
            target_details.append({
                'x_label': current_method_label,
                'value': tv_float,
                'percentile': percentile,
                # 'runs_min': np.min(runs_for_percentile) if len(runs_for_percentile) > 0 else np.nan, # Not strictly needed
                # 'runs_max': np.max(runs_for_percentile) if len(runs_for_percentile) > 0 else np.nan  # Not strictly needed
            })
            current_float = percentile
        else:
            current_float = 0
        if overall_data_tissue is not None:
            if f"query_{query_dataset}_atlas_{current_atlas_dataset}" in overall_data_tissue:
                overall_data_tissue[f"query_{query_dataset}_atlas_{current_atlas_dataset}"].update(
                    {current_method_label: current_float})
            else:
                overall_data_tissue[f"query_{query_dataset}_atlas_{current_atlas_dataset}"] = {
                    current_method_label: current_float
                }

    if not plot_data_list:
        print(f"No data to plot for {query_dataset} across all methods.")
        ax.text(0.5, 0.5, "No data available for any method", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Accuracy Comparison for {query_dataset} ({tissue})", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        df_plot = pd.DataFrame(plot_data_list)
        df_plot['method'] = pd.Categorical(df_plot['method'], categories=method_names_for_plot, ordered=True)

        sns.boxplot(x='method', y='accuracy', hue="method", data=df_plot, ax=ax, order=method_names_for_plot,
                    palette="Set2", width=0.4, dodge=False,
                    showfliers=True)  # Ensure outliers are shown by default for testing

        plotted_target_legend = False
        target_line_half_width = 0.2

        max_y_for_texts_and_targets = -np.inf  # Keep track of highest point needed for text anchors and targets

        for target_info in target_details:
            try:
                x_pos_idx = method_names_for_plot.index(target_info['x_label'])
            except ValueError:
                print(
                    f"Warning: Method label '{target_info['x_label']}' for target not in plotted methods. Skipping target."
                )
                continue

            label_for_legend = None
            # if not plotted_target_legend and not np.isnan(target_info['value']):
            #     label_for_legend = 'Atlas Value'
            #     plotted_target_legend = True

            if not np.isnan(target_info['value']):
                ax.hlines(y=target_info['value'], xmin=x_pos_idx - target_line_half_width,
                          xmax=x_pos_idx + target_line_half_width, colors='red', linestyles='--', linewidth=2,
                          label=label_for_legend, zorder=5)
                max_y_for_texts_and_targets = max(max_y_for_texts_and_targets, target_info['value'])

            current_runs = all_runs_data_for_methods.get(target_info['x_label'])
            text_anchor_y = -np.inf  # Initialize

            if current_runs is None or len(current_runs) == 0:
                if not np.isnan(target_info['value']):
                    text_anchor_y = target_info['value']
                else:  # No runs and no target value to anchor to, skip annotation
                    continue
            else:
                q1, q3 = np.percentile(current_runs, [25, 75])
                iqr = q3 - q1
                upper_whisker_limit = q3 + 1.5 * iqr

                values_within_whisker = current_runs[current_runs <= upper_whisker_limit]
                top_of_whisker_actual = np.max(values_within_whisker) if len(values_within_whisker) > 0 else q3

                outliers = current_runs[current_runs > upper_whisker_limit]
                max_outlier_y = np.max(outliers) if len(outliers) > 0 else -np.inf

                max_boxplot_element_y = max(top_of_whisker_actual, max_outlier_y)

                # Anchor Y is above the highest boxplot element OR the target line, whichever is higher
                text_anchor_y = max_boxplot_element_y
                if not np.isnan(target_info['value']):
                    text_anchor_y = max(text_anchor_y, target_info['value'])

            if np.isinf(text_anchor_y):  # Should not happen if current_runs or target_info['value'] is valid
                print(f"Warning: Could not determine text_anchor_y for {target_info['x_label']}. Skipping annotation.")
                continue

            max_y_for_texts_and_targets = max(max_y_for_texts_and_targets, text_anchor_y)

            percentile_text = f"({target_info['percentile']:.1f}%)" if not np.isnan(target_info['percentile']) else ""
            # Only display target value text if it's valid
            value_text = f"{target_info['value']:.4f}" if not np.isnan(target_info['value']) else "N/A"

            text_to_display_parts = []
            if not np.isnan(target_info['value']):
                text_to_display_parts.append(value_text)
            if percentile_text:  # Only add percentile if it's calculated
                text_to_display_parts.append(percentile_text)

            text_to_display = "\n".join(text_to_display_parts)

            if text_to_display:  # Only annotate if there's something to show
                ax.annotate(
                    text_to_display,
                    xy=(x_pos_idx, text_anchor_y),
                    xytext=(0, 7),  # Increased offset to 7 points
                    textcoords="offset points",
                    color='red',
                    ha='center',
                    va='bottom',
                    size=11,
                    zorder=10)  # Slightly smaller font
        if query_dataset is not None:
            ax.set_title(f"{query_dataset} ({tissue})", fontsize=8, y=1.28)  # y slightly lower
        ax.set_ylabel('Accuracy', fontsize=13)
        ax.set_xlabel(f"Methods (Atlas: {atlas_dataset_for_label})", fontsize=8)

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=13)
        ax.tick_params(axis='y', labelsize=13)

        if plotted_target_legend:
            ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.01, 1.15))  # Adjusted legend

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Adjust Y-axis limits
        current_ymin, current_ymax = ax.get_ylim()
        # Add some padding based on the data range, or a fixed amount if range is tiny
        data_range = current_ymax - current_ymin
        padding = data_range * 0.05  # 5% of current range as padding above the highest text anchor
        if padding < 0.01 and data_range > 0:  # Minimum padding if range is very small but not zero
            padding = 0.01
        elif data_range == 0:  # If all data is flat
            padding = 0.05  # Arbitrary padding

        # Ensure max_y_for_texts_and_targets is valid before using
        if not np.isinf(max_y_for_texts_and_targets):
            new_ymax = max_y_for_texts_and_targets + padding
            ax.set_ylim(bottom=current_ymin, top=max(current_ymax, new_ymax))  # Don't shrink, only expand
        else:  # if no text/targets were plotted, use a default small padding
            ax.set_ylim(bottom=current_ymin, top=current_ymax + padding)

    # if not method_names_for_plot: # If loop was skipped for all methods
    # ax.set_xlabel("")
    ax.set_xticks([])

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # rect to give space for title/legend and x-labels

    if not isinstance(SIMILARITYDIR, Path): base_dir = Path(SIMILARITYDIR)
    else: base_dir = SIMILARITYDIR

    path_parts = ["data", "imgs", "imgs_v2"]
    path_parts.append(str(tissue))
    result_path_dir = base_dir.joinpath(*path_parts)

    os.makedirs(result_path_dir, exist_ok=True)
    result_path_file = result_path_dir / f"{query_dataset}.pdf"
    if save:
        plt.savefig(result_path_file, dpi=300, format='pdf')
        print(f"Saved plot to {result_path_file}")
    # plt.close(fig)
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tissue", type=str, default="brain")
    args = parser.parse_args()
    tissue = args.tissue
    with open(SIMILARITYDIR / "data/overall/overall_data.json") as f:
        overall_data = json.load(f)
    overall_data_tissue = overall_data.get(tissue, {})
    feature_name = "wasserstein"
    set_seed(42)
    # conf_data = pd.read_csv(f"results/{tissue}_result.csv", index_col=0)
    conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
    query_datasets = list(conf_data[conf_data["queryed"] == True]["dataset_id"])
    for exclude_dataset in exclude_data.get(tissue, []):
        if exclude_dataset in query_datasets:
            query_datasets.remove(exclude_dataset)
    ans = get_ans(query_datasets, tissue, exclude_data=exclude_data)
    methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
    plt.style.use("default")
    import json
    for query_dataset, data in ans.items():
        plot_combined_methods(data, query_dataset, methods, tissue, feature_name, conf_data,
                              overall_data_tissue=overall_data_tissue)
    with open(SIMILARITYDIR / f"data/overall/overall_data.json", "w") as f:
        overall_data[tissue] = overall_data_tissue
        json.dump(overall_data, f, indent=4)
