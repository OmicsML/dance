"""Calculate optimal weights for combining similarity metrics in cell type annotation.

This script analyzes different similarity metrics (like Wasserstein, Hausdorff, etc.) and metadata similarity
to find optimal weights that minimize the total rank of correct cell type predictions across multiple datasets.

The script:
1. Loads similarity scores from Excel files
2. Computes rankings for different cell type annotation methods
3. Finds optimal weights (w1, w2) for combining feature-based and metadata-based similarity
4. Outputs the best performing feature and its corresponding weight

Returns
-------
DataFrame
    Results containing feature names, weights, and corresponding total ranks

"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from dance.settings import SIMILARITYDIR

sys.path.append(str(SIMILARITYDIR))  # 添加父目录到路径
print(sys.path)
from similarity.process_tissue_similarity_matrices import convert_to_complex
from visualization.visualize_atlas_performance import get_runs

from dance.utils import set_seed, try_import


def get_ans():
    """Load similarity scores from Excel files for each dataset.

    Returns
    -------
    dict
        Dictionary mapping dataset IDs to their similarity score DataFrames

    """
    ans = {}
    for query_dataset in query_datasets:
        if query_dataset in exclude_dataset:
            continue
        data = pd.read_excel(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx", sheet_name=query_dataset[:4],
                             index_col=0)
        ans[query_dataset] = data
    return ans


def get_rank(reduce_error, in_query):
    """Calculate rankings for each cell type annotation method.

    Updates the input DataFrames with rank columns for each method, where lower ranks
    indicate better performance.

    """

    def query_rank(x, test_accs, reduce_error):
        if pd.isna(x):
            if reduce_error:
                return 10000
            else:
                x = -0.01
        rank = sum(1 for test_acc in test_accs if test_acc > x) + 1
        return rank

    if in_query:
        for query_dataset, data in ans.items():
            for method in methods:
                rank_col = 'rank_' + method
                data.loc[rank_col, :] = data.loc[method, :].apply(
                    lambda x: query_rank(x, get_runs(conf_data, query_dataset, method), reduce_error))
    else:
        for query_dataset, data in ans.items():
            for method in methods:
                rank_col = 'rank_' + method
                # if method not in data.index:
                #     data.loc[rank_col, :] = 10000
                #     continue
                if reduce_error:
                    data.loc[rank_col, :] = data.loc[method, :].rank(ascending=False, method='min', na_option='keep')
                    data.loc[rank_col, :] = data.loc[rank_col, :].fillna(10000)
                else:
                    data.loc[rank_col, :] = data.loc[method, :].rank(ascending=False, method='min', na_option='bottom')


def objective(w1, feature_name):
    """Calculate total rank score for given weights and feature.

    Parameters
    ----------
    w1 : float
        Weight for the feature-based similarity (0-1)
    feature_name : str
        Name of the similarity feature to evaluate

    Returns
    -------
    float
        Total rank score (lower is better)

    """
    w2 = 1 - w1
    total_rank = 0
    for query_dataset, data in ans.items():
        df_A = data.copy()
        # if feature_name == "bures":
        df_A.loc[feature_name, :] = df_A.loc[feature_name, :].apply(convert_to_complex)
        # print(df_A.loc[feature_name, :])
        df_A.loc['score_similarity', :] = w1 * df_A.loc[feature_name, :].values.astype(float) + w2 * df_A.loc[
            'metadata_sim', :].values.astype(float)
        if df_A.loc['score_similarity', :].isna().any():
            pass
        df_A.loc['score_similarity', :] = df_A.loc['score_similarity', :].fillna(0)
        max_idx = df_A.loc['score_similarity', :].idxmax()
        max_B = df_A.loc[:, max_idx]
        ranks = []
        for method in methods:
            ranks.append(max_B.loc['rank_' + method])
        total_rank += np.sum(ranks)
    return total_rank


if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    wandb = try_import("wandb")
    entity = "xzy11632"
    project = "dance-dev"
    # query_datasets = [
    #     "c7775e88-49bf-4ba2-a03b-93f00447c958",
    #     "456e8b9b-f872-488b-871d-94534090a865",
    #     "738942eb-ac72-44ff-a64b-8943b5ecd8d9",
    #     # "a5d95a42-0137-496f-8a60-101e17f263c8",
    #     "71be997d-ff75-41b9-8a9f-1288c865f921"
    # ]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tissue", type=str, default="heart")
    parser.add_argument("--reduce_error", action="store_true")
    parser.add_argument("--in_query", action="store_true")

    args = parser.parse_args()
    tissue = args.tissue
    reduce_error = args.reduce_error
    in_query = args.in_query
    set_seed(42)
    # conf_data = pd.read_csv(f"results/{tissue}_result.csv", index_col=0)
    conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
    query_datasets = list(conf_data[conf_data["queryed"] == True]["dataset_id"])
    methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
    feature_names = ["wasserstein", "Hausdorff", "chamfer", "energy", "sinkhorn2", "bures", "spectral", "mmd"]
    import json
    with open(SIMILARITYDIR / "configs/exclude_dataset.json", encoding='utf-8') as f:
        exclude_dataset_json = json.load(f)
        exclude_dataset = exclude_dataset_json[tissue] if tissue in exclude_dataset_json else []

    ans = get_ans()
    get_rank(reduce_error=reduce_error, in_query=in_query)
    all_results = []
    # for query_dataset, data in ans.items():
    #     data.to_csv(f"ranks/{tissue}/{query_dataset}_rank.csv")
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
    results_df.to_csv(f"temp/{tissue}_results_df.csv")
    best_result = results_df.loc[results_df['total_rank'].idxmin()]

    print('Best similarity feature:', best_result['feature_name'])
    print('Best w1:', best_result['w1'])
    print('Corresponding total rank:', best_result['total_rank'])

    import json
    import os
    result_path = SIMILARITYDIR / f"data/similarity_weights_results/{'reduce_error_' if reduce_error else ''}{'in_query_' if in_query else ''}sim_dict.json"
    if not os.path.exists(result_path):
        sim_dict = {}
    else:
        with open(result_path, encoding='utf-8') as f:
            sim_dict = json.load(f)
    sim_dict[tissue] = {
        "feature_name": best_result['feature_name'],
        "weight1": best_result['w1'],
        "total rank": int(best_result['total_rank'])
    }  # 添加单个值
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(sim_dict, f, indent=4, ensure_ascii=False)
