import argparse
import os
import re
import sys
from math import e, pi
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

exclude_data = {
    "blood": ["a5d95a42-0137-496f-8a60-101e17f263c8"],
    "kidney": ["0b75c598-0893-4216-afe8-5414cab7739d"],
    "pancreas": [
        "78f10833-3e61-4fad-96c9-4bbd4f14bdfa", "9c4c8515-8f82-4c72-b0c6-f87647b00bbe",
        "5a11f879-d1ef-458a-910c-9b0bdfca5ebf"
    ],
    "brain": [
        "07760522-707a-4a1c-8891-dbd1226d6b27", "146216e1-ec30-4fee-a1fb-25defe801e2d",
        "700aed19-c16e-4ba8-9191-07da098a8626", "9813a1d4-d107-459e-9b2e-7687be935f69"
    ],
    "heart": [
        "f7995301-7551-4e1d-8396-ffe3c9497ace", "1062c0f2-2a44-4cf9-a7c8-b5ed58b4728d",
        "1252c5fb-945f-42d6-b1a8-8a3bd864384b", "83b5e943-a1d5-4164-b3f2-f7a37f01b524",
        "bdf69f8d-5a96-4d6f-a9f5-9ee0e33597b7", "1009f384-b12d-448e-ba9f-1b7d2ecfbb4e",
        "f75f2ff4-2884-4c2d-b375-70de37a34507", "97a17473-e2b1-4f31-a544-44a60773e2dd"
    ]
}


def get_atlas_ans(query_dataset, method, feature_name="wasserstein", data=None):
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
    in_query = True
    # with open(
    #         SIMILARITYDIR /
    #         f"data/similarity_weights_results/{'reduce_error_' if reduce_error else ''}{'in_query_' if in_query else ''}sim_dict.json",
    #         encoding='utf-8') as f:
    #     sim_dict = json.load(f)
    # feature_name = sim_dict[tissue]["feature_name"]
    # data = pd.read_excel(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx", sheet_name=query_dataset[:4],
    #                      index_col=0)
    # weight1 = 1.0  # Weight for feature-based similarity
    data.loc[feature_name, :] = data.loc[feature_name, :].apply(convert_to_complex)
    weighted_sum = (data.loc[feature_name, :]).astype(float)
    atlas_dataset_res = weighted_sum.idxmax()  # Get most similar dataset
    max_value = weighted_sum.max()
    if method in data.index:
        return data.loc[method, atlas_dataset_res], atlas_dataset_res
    else:
        return 0, "null"


def get_ans(query_datasets, tissue, exclude_data):
    """Load similarity scores from Excel files for each dataset.

    Returns
    -------
    dict
        Dictionary mapping dataset IDs to their similarity score DataFrames

    """
    ans = {}
    for query_dataset in query_datasets:
        data = pd.read_excel(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx", sheet_name=query_dataset[:4],
                             index_col=0)
        ans[query_dataset] = data.drop(exclude_data.get(tissue, []), axis=1, errors='ignore')
    return ans


def get_atlas_acc(ans, methods):
    for query_dataset, data in ans.items():
        for method in methods:
            acc_col = 'acc_' + method
            for column in data.columns:
                if pd.isna(data.loc[method, column]):
                    print(f"Warning: {method} has NaN for {query_dataset} in {column}. Setting to 0.")
            data.loc[acc_col, :] = data.loc[method, :].fillna(0)
    for query_dataset, data in ans.items():
        if "sum_acc" not in data.index:
            data.loc['sum_acc', :] = 0
        for method in methods:
            acc_col = 'acc_' + method
            data.loc['sum_acc', :] += data.loc[acc_col, :]
    for query_dataset, data in ans.items():
        data.loc["average_acc"] = data.loc["sum_acc"].astype(float) * 0.25


if __name__ == "__main__":
    tissues = ["blood", "brain", "heart", "intestine", "kidney", "lung", "pancreas"]
    for tissue in tissues:
        set_seed(42)
        conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
        query_datasets = list(conf_data[conf_data["queryed"] == True]["dataset_id"])
        if tissue in exclude_data:
            for exclude_dataset in exclude_data[tissue]:
                if exclude_dataset in query_datasets:
                    query_datasets.remove(exclude_dataset)
        plt.style.use("default")
        ans = get_ans(query_datasets, tissue, exclude_data)
        methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
        import json
        feature_names = ["wasserstein", "Hausdorff", "chamfer", "energy", "sinkhorn2", "bures", "spectral", "mmd"]
        feature_names.append("metadata_sim")
        feature_names.append("average_acc")
        acc_query = []
        get_atlas_acc(ans, methods)
        for feature_name in feature_names:
            # if feature_name != "Hausdorff":
            #     continue  # Skip if feature_name is not "Hausdorff"
            for query_dataset, data in ans.items():
                methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
                for i, method_key in enumerate(methods):  # Renamed 'method' to 'method_key'
                    target_value_str, current_atlas_dataset = get_atlas_ans(query_dataset, method_key,
                                                                            feature_name=feature_name,
                                                                            data=data)  # only for current_atlas_dataset
                acc_query_single = {}
                acc_query_single["query_dataset"] = query_dataset
                acc_query_single["current_atlas_dataset"] = current_atlas_dataset
                acc_query_single["average_acc"] = data.loc["average_acc", current_atlas_dataset]
                acc_query_single["feature_name"] = feature_name
                acc_query.append(acc_query_single)
                df_sim = data.loc[feature_names, :].T.applymap(convert_to_complex)

        pd.DataFrame(acc_query).set_index(["query_dataset", "current_atlas_dataset"
                                           ]).to_csv(SIMILARITYDIR / f"data/atlas_accs/{tissue}_atlas_acc.csv")
