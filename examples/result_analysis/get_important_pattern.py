# metric_name = "test_acc"
# ascending = False
import argparse
import itertools
import pathlib
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from networkx import parse_adjlist
from scipy import cluster, stats


#TODO need to sync all files or get sweep,not file
#asceding need to think
#负向的pattern，换一下顺序就可以吧
def get_important_pattern(test_accs, ascending, vis=True, alpha=0.05, title=""):

    if vis:
        fig = plt.figure(figsize=(12, 4))
        sns.boxplot(data=test_accs)
        plt.xticks(list(range(len(test_accs))), [f"{i}" for i in range(len(test_accs))])
        plt.title(title)
        plt.show()
    _, p_value = stats.kruskal(*test_accs)
    if p_value < alpha:
        medians = [np.median(group) for group in test_accs]
        data = test_accs
        p_values_matrix = sp.posthoc_dunn(a=data, p_adjust="bonferroni")
        sorted_indices = np.argsort(np.argsort([-x for x in medians] if ascending else medians))
        ranks = {
            index: {
                "rank": rank,
                "before": None,
                "after": [],
                "real_rank": rank
            }
            for index, rank in enumerate(sorted_indices)
        }
        for (rank1, rank2) in combinations(range(max(sorted_indices) + 1), 2):
            for idx1 in [index for index, value in ranks.items() if value["rank"] == rank1]:
                for idx2 in [index for index, value in ranks.items() if value["rank"] == rank2]:
                    if p_values_matrix.iloc[idx1, idx2] > alpha:
                        if ranks[idx2]["before"] is None:
                            ranks[idx1]["after"].append(idx2)
                            ranks[idx2]["before"] = idx1

        def change_real_rank(rank_item, real_rank):
            rank_item["real_rank"] = real_rank
            for idx in rank_item["after"]:
                change_real_rank(ranks[idx], real_rank)

        for rank_item in ranks.values():
            if rank_item["before"] is None:
                for idx in rank_item["after"]:
                    change_real_rank(ranks[idx], rank_item["real_rank"])
        return [v["real_rank"] for k, v in ranks.items()]
    else:
        if vis:
            print("No significant differences found between the groups.")
        return []


def replace_nan_in_2d(lst):
    return [[np.nan if item == 'NaN' else item for item in sublist] for sublist in lst]


def are_all_elements_same_direct(list_2d):
    first_element = None
    for sublist in list_2d:
        for element in sublist:
            if first_element is None:
                first_element = element
            elif element != first_element:
                return False
    return True if first_element is not None else True


def get_frequent_itemsets(step2_data, metric_name, ascending, threshold_per=0.1):
    threshold = int(len(step2_data) * threshold_per)
    step2_data.loc[:, metric_name] = step2_data.loc[:, metric_name].astype(float)
    df_sorted = step2_data.sort_values(metric_name, ascending=ascending)
    top_10_percent = df_sorted.head(threshold)
    columns = sorted([col for col in step2_data.columns if col.startswith("pipeline")])
    transactions = top_10_percent[columns].values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    # print(frequent_itemsets)
    # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    return [tuple(a) for a in frequent_itemsets["itemsets"]]


# def get_significant_top_n_zscore(data, n=3, threshold=1.0, ascending=False):
#     if not data:
#         return []

#     n = max(1, n)

#     mean = np.mean(data)
#     std = np.std(data)

#     if std == 0:
#         return sorted(data, reverse=not ascending)[:n]

#     z_scores = [(x, (x - mean) / std) for x in data]

#     significant_values = [x for x, z in z_scores if z > threshold]

#     significant_values_sorted = sorted(significant_values, reverse=not ascending)

#     if len(significant_values_sorted) < n:
#         remaining = sorted(data, reverse=not ascending)[:n - len(significant_values_sorted)]
#         significant_values_sorted.extend(remaining)

#     return significant_values_sorted[:n]


def get_com_all(step2_data, metric_name, ascending, vis=True, alpha=0.05):
    ans_all = []
    columns = sorted([col for col in step2_data.columns if col.startswith("pipeline")])
    test_accs = []
    test_acc_names = []
    for r in range(1, len(columns)):  #全流程的单独处理
        for com in itertools.combinations(columns, r):
            test_accs_arrays = []
            groups = step2_data.groupby(by=list(com))
            if len(groups) == 1:
                continue
            for g in groups:
                test_accs_arrays.append({"name": g[0], metric_name: list(g[1][metric_name])})
            test_accs += [i[metric_name] for i in test_accs_arrays]
            test_acc_names += [i["name"] for i in test_accs_arrays]
            # if are_all_elements_same_direct(test_accs):
            #     continue
    test_accs = replace_nan_in_2d(test_accs)
    final_ranks = get_important_pattern(test_accs, ascending, alpha=alpha, title=" ".join(list(com)), vis=vis)
    if len(final_ranks) > 0:  #TODO maybe need to think ascending
        max_rank = max(final_ranks)
        max_rank_count = final_ranks.count(max_rank)
        if max_rank_count < len(final_ranks) / 2:
            for index, (test_acc_name, rank) in enumerate(zip(test_acc_names, final_ranks)):
                if rank == max_rank:
                    if vis:
                        print(f"index={index},name={test_acc_name},rank={rank}")
                    ans_all.append(test_acc_name if isinstance(test_acc_name, tuple) else (test_acc_name, ))
    return ans_all


def summary_pattern(data_path, metric_name, ascending, alpha=0.05, vis=False):
    step2_origin_data = pd.read_csv(data_path)
    step2_data = step2_origin_data.dropna()
    com_ans = get_com_all(step2_data, metric_name, ascending, vis=vis, alpha=alpha)
    apr_ans = get_frequent_itemsets(step2_data, metric_name, ascending)
    return list(set(com_ans) & set(apr_ans))


# def list_files(directory,file_name="best_test_acc.csv",save_path="summary_file"):
#     ans=[]
#     path = Path(directory)
#     for file_path in path.rglob('*'):
#         if file_path.is_file():
#             if file_path.name==file_name:
#                 algorithm,dataset=file_path.relative_to(directory).parts[:2]
#                 ans.append({"algorithm":algorithm,"dataset":dataset,"summary_pattern":summary_pattern(file_path)})
#     pd.DataFrame(ans).to_csv(save_path)
def list_files(directories, metric_name, ascending, file_name="best_test_acc.csv", alpha=0.05, vis=False):
    ans_all = []
    for directory in directories:
        path = Path(directory)
        for file_path in path.rglob('*'):
            if file_path.is_file():
                if file_path.name == file_name:
                    print(file_path)
                    dataset = file_path.parent
                    method = file_path.parent.parent
                    ans = summary_pattern(file_path, metric_name, ascending, alpha=alpha, vis=vis)
                    with open(Path(file_path.parent.resolve(), "pipeline_summary_pattern.txt"), 'w') as f:
                        f.write(str(ans))
                    ans_all.append({"dataset": dataset, "method": method, "ans": ans})
    return ans_all


if __name__ == "__main__":
    directories = []
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("task", default="cluster")
    parser.add_argument("metric_name", default="acc")
    parser.add_argument("ascending", default=False)
    args = parser.parse_args()
    task = args.task
    metric_name = args.metric_name
    ascending = args.ascending
    file_root = Path(__file__).resolve().parent.parent / "tuning"
    for path in file_root.iterdir():
        if path.is_dir():
            if str(path.name).startswith(task):
                directories.append(path)
    ans_all = list_files(directories, metric_name, ascending)
    df = pd.DataFrame(ans_all)
    pivot_df = df.pivot(index="dataset", columns="method", values="ans")
    pivot_df.to_csv(f"{task}_pattern.csv")

    # print(summary_pattern("/home/zyxing/dance/examples/tuning/cta_actinn/328_138/results/pipeline/best_test_acc.csv",alpha=0.3,vis=True))
