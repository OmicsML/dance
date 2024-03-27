import itertools
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats

metric_name = "acc"


def get_important_pattern(test_accs, vis=True, alpha=0.8, title=""):
    medians = [np.median(group) for group in test_accs]
    _, p_value = stats.kruskal(*test_accs)
    if vis:
        fig = plt.figure(figsize=(12, 4))
        sns.boxplot(data=test_accs)
        plt.xticks(list(range(len(test_accs))), [f"{i}" for i in range(len(test_accs))])
        plt.title(title)
        plt.show()
    if p_value < alpha:
        data = test_accs
        p_values_matrix = sp.posthoc_dunn(a=data)
        sorted_indices = np.argsort(medians)
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
        print("No significant differences found between the groups.")
        return []


def get_com(step2_data, r=2, alpha=0.8, columns=None):
    ans = []
    for com in itertools.combinations(columns, r):
        test_accs_arrays = []
        for g in step2_data.groupby(by=list(com)):
            test_accs_arrays.append({"name": g[0], metric_name: list(g[1][metric_name])})
        test_accs = [i[metric_name] for i in test_accs_arrays]
        test_acc_names = [i["name"] for i in test_accs_arrays]
        final_ranks = get_important_pattern(test_accs, alpha=alpha, title=" ".join(list(com)))
        if len(final_ranks) > 0:
            max_rank = max(final_ranks)
            max_rank_count = final_ranks.count(max_rank)
            if max_rank_count < len(final_ranks) / 2:
                for index, (test_acc_name, rank) in enumerate(zip(test_acc_names, final_ranks)):
                    if rank == max_rank:
                        print(f"index={index},name={test_acc_name},rank={rank}")
                        ans.append(test_acc_name if isinstance(test_acc_name, tuple) else (test_acc_name, ))
    return ans


def get_frequent_itemsets(step2_data, threshold_per=0.1):
    threshold = int(len(step2_data) * threshold_per)
    df_sorted = step2_data.sort_values(metric_name, ascending=False)
    top_10_percent = df_sorted.head(threshold)
    columns = sorted([col for col in step2_data.columns if col.startswith("pipeline")])
    transactions = top_10_percent[columns].values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    print(frequent_itemsets)
    # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    return [tuple(a) for a in frequent_itemsets["itemsets"]]


def get_com_all(step2_data):
    ans = []
    columns = sorted([col for col in step2_data.columns if col.startswith("pipeline")])
    for i in range(1, len(columns)):
        ans += get_com(step2_data, i, columns=columns)
    return ans


def summary_pattern(data_path):
    step2_origin_data = pd.read_csv(data_path)
    step2_data = step2_origin_data.dropna()
    print(step2_data.shape)
    com_ans = get_com_all(step2_data)
    apr_ans = get_frequent_itemsets(step2_data)
    print("----------------")
    print(com_ans)
    print(apr_ans)
    print(list(set(com_ans) & set(apr_ans)))


if __name__ == "__main__":
    summary_pattern("/home/zyxing/dance/examples/tuning/cluster_graphsc/10X_PBMC/results/pipeline/best_test_acc.csv")
