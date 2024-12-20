import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dance.utils import try_import

wandb = try_import("wandb")
entity = "xzy11632"
project = "dance-dev"
query_datasets = [
    "c7775e88-49bf-4ba2-a03b-93f00447c958",
    "456e8b9b-f872-488b-871d-94534090a865",
    "738942eb-ac72-44ff-a64b-8943b5ecd8d9",
    # "a5d95a42-0137-496f-8a60-101e17f263c8",
    "71be997d-ff75-41b9-8a9f-1288c865f921"
]
methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
file_root = Path(__file__).resolve().parent
feature_names = ["wasserstein", "Hausdorff", "chamfer", "energy", "sinkhorn2", "bures", "spectral", "mmd"]


def get_ans():
    ans = {}
    for query_dataset in query_datasets:
        data = pd.read_excel(file_root / "Blood_similarity.xlsx", sheet_name=query_dataset[:4], index_col=0)
        ans[query_dataset] = data
    return ans


def get_rank():
    for query_dataset, data in ans.items():
        for method in methods:
            rank_col = 'rank_' + method
            data.loc[rank_col, :] = data.loc[method, :].rank(ascending=False, method='min', na_option='bottom')
            # data.loc[rank_col,:] = data.loc[rank_col,:].fillna(10000)


def convert_to_complex(s):
    if isinstance(s, float) or isinstance(s, int):
        return float(s)
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return np.nan


def objective(w1, feature_name):
    w2 = 1 - w1
    total_rank = 0
    for query_dataset, data in ans.items():
        df_A = data.copy()
        if feature_name == "bures":
            df_A.loc[feature_name, :] = df_A.loc[feature_name, :].apply(convert_to_complex)
            df_A.loc[feature_name, :] = df_A.loc[feature_name, :].apply(lambda x: x.real
                                                                        if isinstance(x, complex) else np.nan)
            print(df_A.loc[feature_name, :])
        df_A.loc['score_similarity', :] = w1 * df_A.loc[feature_name, :].values.astype(float) + w2 * df_A.loc[
            'metadata_sim', :].values.astype(float)
        # df_A.loc['score_similarity',:]= df_A.loc['score_similarity',:].fillna(0)
        max_idx = df_A.loc['score_similarity', :].idxmax()
        max_B = df_A.loc[:, max_idx]
        ranks = []
        for method in methods:
            ranks.append(max_B.loc['rank_' + method])
        total_rank += np.sum(ranks)
    return total_rank


ans = get_ans()
get_rank()
all_results = []
for query_dataset, data in ans.items():
    data.to_csv(f"ranks/{query_dataset}_rank.csv")
for feature_name in feature_names:
    w1_values = np.linspace(0, 1, 101)
    results = []
    for w1 in w1_values:
        total_rank = objective(w1, feature_name)
        results.append({'feature_name': feature_name, 'w1': w1, 'total_rank': total_rank})
    all_results.extend(results)
# for w1 in w1_values:
#     total_rank = objective(w1)
#     results.append({'w1': w1, 'total_rank': total_rank})

results_df = pd.DataFrame(all_results)
results_df.to_csv("temp/results_df.csv")
best_result = results_df.loc[results_df['total_rank'].idxmin()]

print('最佳相似性特征:', best_result['feature_name'])
print('最佳 w1:', best_result['w1'])
print('对应的总排名:', best_result['total_rank'])
