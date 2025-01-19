import argparse
import json
import os
import pathlib

import pandas as pd
import scanpy as sc
from dotenv import load_dotenv

from dance.atlas.data_dropbox_upload import get_ans, get_data

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()

    args = argparse.ArgumentParser()
    args.add_argument("--maindir", type=str)
    args.add_argument("--filedir", type=str)
    args.add_argument("--tissues", type=str, nargs="+")
    args.add_argument("--dropbox_dest_path", type=str)
    args = args.parse_args()

    # 从环境变量获取 access_token
    ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')
    if not ACCESS_TOKEN:
        raise ValueError("未找到 DROPBOX_ACCESS_TOKEN 环境变量！\n"
                         "请在 .env 文件中设置 DROPBOX_ACCESS_TOKEN=your_token_here")

    MAINDIR = pathlib.Path(args.maindir)
    FILEDIR = pathlib.Path(args.filedir)
    tissues = args.tissues
    DROPBOX_DEST_PATH = args.dropbox_dest_path

    def get_data(dataset_id, in_atlas=False, large=False):
        """Load h5ad dataset from local path.

        Parameters
        ----------
        dataset_id : str
            Identifier for the dataset
        in_atlas : bool
            Whether dataset is from atlas (True) or query (False)
        large : bool
            Whether dataset is large (>10000 cells) requiring sampling

        Returns
        -------
        AnnData
            Loaded single cell data
        Path
            Local path to the data file

        """
        if large:
            if in_atlas:
                local_path = MAINDIR / f"sampled-10000/{tissue}/{dataset_id}.h5ad"
            else:
                local_path = FILEDIR / f"sampled-10000/{tissue}/{dataset_id}.h5ad"
        else:
            local_path = MAINDIR / f"{tissue}/{dataset_id}.h5ad"
        data = sc.read_h5ad(local_path)
        return data, local_path

    ans_all = []

    with open(FILEDIR / "results/atlas_result.json") as f:
        result = json.load(f)
    with open(FILEDIR / "results/query_result.json") as f:
        query_result = json.load(f)
    for tissue in tissues:
        large_dataset_ids = result[tissue][0]
        small_dataset_ids = result[tissue][1]
        for large_dataset_id in large_dataset_ids:
            data, local_path = get_data(dataset_id=large_dataset_id, in_atlas=True, large=True)
            ans_all.append(
                get_ans(dataset_id=large_dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))
        for small_dataset_id in small_dataset_ids:
            data, local_path = get_data(dataset_id=small_dataset_id, in_atlas=True, large=False)
            ans_all.append(
                get_ans(dataset_id=small_dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))
        large_query_dataset_ids = query_result[tissue][0]
        small_query_dataset_ids = query_result[tissue][1]
        for large_query_dataset_id in large_query_dataset_ids:
            data, local_path = get_data(dataset_id=large_query_dataset_id, in_atlas=False, large=True)
            ans_all.append(
                get_ans(dataset_id=large_query_dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))
        for small_query_dataset_id in small_query_dataset_ids:
            data, local_path = get_data(dataset_id=small_query_dataset_id, in_atlas=False, large=False)
            ans_all.append(
                get_ans(dataset_id=small_query_dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))
    pd.DataFrame(ans_all).set_index("species").to_csv(",".join(tissues) + "scdeeepsort.csv")
