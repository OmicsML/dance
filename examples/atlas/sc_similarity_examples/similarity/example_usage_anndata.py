import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy.sparse import issparse
from torch.utils.data import TensorDataset

from dance.atlas.sc_similarity.anndata_similarity import AnnDataSimilarity, get_anndata
from dance.settings import DANCEDIR, SIMILARITYDIR
from dance.utils import set_seed

# target_files = [
#     "01209dce-3575-4bed-b1df-129f57fbc031", "055ca631-6ffb-40de-815e-b931e10718c0",
#     "2a498ace-872a-4935-984b-1afa70fd9886", "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf",
#     "3faad104-2ab8-4434-816d-474d8d2641db", "471647b3-04fe-4c76-8372-3264feb950e8",
#     "4c4cd77c-8fee-4836-9145-16562a8782fe", "84230ea4-998d-4aa8-8456-81dd54ce23af",
#     "8a554710-08bc-4005-87cd-da9675bdc2e7", "ae29ebd0-1973-40a4-a6af-d15a5f77a80f",
#     "bc260987-8ee5-4b6e-8773-72805166b3f7", "bc2a7b3d-f04e-477e-96c9-9d5367d5425c",
#     "d3566d6a-a455-4a15-980f-45eb29114cab", "d9b4bc69-ed90-4f5f-99b2-61b0681ba436",
#     "eeacb0c1-2217-4cf6-b8ce-1f0fedf1b569"
# ]
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tissue", type=str, default="pancreas")
parser.add_argument("--data_dir", default=DANCEDIR / f"examples/tuning/temp_data")
args = parser.parse_args()

data_dir = args.data_dir
set_seed(42)
tissue = args.tissue
# conf_data = pd.read_csv(f"results/{tissue}_result.csv", index_col=0)
conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
atlas_datasets = list(conf_data[conf_data["queryed"] == False]["dataset_id"])
query_datasets = list(conf_data[conf_data["queryed"] == True]["dataset_id"])


class CustomEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super().default(obj)


def dataset_from_anndata(adata: AnnData, label_key: str = 'cell_type', classes=None):
    """Convert AnnData object to PyTorch TensorDataset.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    label_key : str, default='cell_type'
        Column name in adata.obs containing cell type labels
    classes : list, optional
        Predefined class labels. If None, will be inferred from data

    Returns
    -------
    TensorDataset
        PyTorch dataset with features and labels

    """
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X_tensor = torch.from_numpy(X).float()
    Y = adata.obs[label_key].values
    if pd.api.types.is_numeric_dtype(Y):
        targets = torch.LongTensor(Y)
        if classes is None:
            classes = sorted(np.unique(Y))
    else:
        unique_classes = sorted(np.unique(Y))
        # class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # Y_encoded = np.array([class_to_idx[cls] for cls in Y])
        targets = torch.LongTensor(Y.codes)
        if classes is None:
            classes = unique_classes
    ds = TensorDataset(X_tensor, targets)
    ds.targets = targets
    ds.classes = classes
    return ds


# def run_test_otdd():
#     data_root = "/home/zyxing/dance/examples/tuning/temp_data/train/human"
#     for target_file in target_files:
#         source_data = sc.read_h5ad(f"{data_root}/human_{tissue.capitalize()}{source_file}_data.h5ad")
#         target_data = sc.read_h5ad(f"{data_root}/human_{tissue.capitalize()}{target_file}_data.h5ad")
#         source_ds = dataset_from_anndata(source_data)
#         target_ds = dataset_from_anndata(target_data)
#         dist = DatasetDistance(source_ds, target_ds)
#         dist.distance()


def run_test_case(source_file):
    """Calculate similarity matrices between source and target datasets.

    Parameters
    ----------
    source_file : str
        Name of the source dataset file

    Returns
    -------
    pandas.DataFrame
        Similarity scores for different metrics

    """
    ans = {}
    source_data = get_anndata(train_dataset=[f"{source_file}"], data_dir=data_dir, tissue=tissue.capitalize())

    for target_file in atlas_datasets:
        # source_data=sc.read_h5ad(f"{data_root}/{source_file}.h5ad")
        # target_data=sc.read_h5ad(f"{data_root}/{target_file}.h5ad")
        target_data = get_anndata(train_dataset=[f"{target_file}"], data_dir=data_dir, tissue=tissue.capitalize())

        # Initialize similarity calculator with multiple metrics
        similarity_calculator = AnnDataSimilarity(
            adata1=source_data, adata2=target_data, sample_size=10, init_random_state=42, n_runs=1,
            ground_truth_conf_path=SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", adata1_name=source_file,
            adata2_name=target_file, tissue=tissue)

        # Calculate similarity using multiple methods
        ans[target_file] = similarity_calculator.get_similarity_matrix_A2B(methods=[
            "wasserstein",
            # "Hausdorff",
            # "chamfer",
            # "energy",
            # "sinkhorn2",
            # "bures",
            # "spectral",
            # "common_genes_num",
            "ground_truth",
            # "mmd",
            # "metadata_sim"
        ])

    # Convert results to DataFrame and save
    ans = pd.DataFrame(ans)
    # ans_to_path = f'sims/{tissue}/sim_{source_file}.csv'
    # os.makedirs(os.path.dirname(ans_to_path), exist_ok=True)
    # ans.to_csv(ans_to_path)
    return ans


start = False
query_data = os.listdir(SIMILARITYDIR / "data/in_atlas_datas" / f"{tissue}")
excel_path = SIMILARITYDIR / f"data/dataset_similarity/{tissue}_similarity.xlsx"
# with pd.ExcelWriter(file_root / f"{tissue}_similarity.xlsx", engine='openpyxl') as writer:
for source_file in query_datasets:
    # if source_file[:4]=='e868':
    #     start=True
    # if not start:
    #     continue
    query_ans = pd.concat([
        pd.read_csv(SIMILARITYDIR / "data/in_atlas_datas" / f"{tissue}" / element, index_col=0)
        for element in query_data if element.split("_")[-3] == source_file
    ])
    rename_dict = {col: col.replace('_from_cache', '') for col in query_ans.columns if '_from_cache' in col}
    query_ans = query_ans.rename(columns=rename_dict)
    ans = run_test_case(source_file)
    merged_df = pd.concat([query_ans, ans], join='inner')
    if os.path.exists(excel_path):
        excel = pd.ExcelFile(excel_path, engine='openpyxl')
        if source_file[:4] in excel.sheet_names:
            # 尝试读取指定的分表
            existing_df = pd.read_excel(SIMILARITYDIR / f"data/dataset_similarity/{tissue}_similarity.xlsx",
                                        sheet_name=source_file[:4], engine="openpyxl", index_col=0)
            # 找出在新数据框中存在但在现有表格中不存在的行
            merged_df = pd.concat([existing_df, merged_df])
            merged_df = merged_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
            # 2. 然后去重，这时应该使用keep='last'而不是subset参数
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            # merged_df = merged_df.drop_duplicates(subset=merged_df.index.name, keep='last')
        excel.close()
    if os.path.exists(excel_path):
        mode = 'a'
        if_sheet_exists = "replace"
    else:
        mode = 'w'
        if_sheet_exists = None
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
        merged_df.to_excel(writer, sheet_name=source_file[:4])
