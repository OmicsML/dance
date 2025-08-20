"""Upload Atlas and Query datasets to Dropbox.

This script handles the upload of single-cell RNA sequencing datasets to Dropbox.
It processes both atlas and query datasets, handling large (>10000 cells) and small datasets separately.
The script reads data from local h5ad files and uploads them to a specified Dropbox location.

Required environment variables:
    DROPBOX_ACCESS_TOKEN: Authentication token for Dropbox API

Usage:
    python upload_data.py --maindir <atlas_dir> --filedir <query_dir>
                         --tissues <tissue_list> --dropbox_dest_path <dest_path>

"""

import argparse
import json
import os
import pathlib

import pandas as pd
import scanpy as sc
from dotenv import load_dotenv

from dance.atlas.data_dropbox_upload import get_ans, get_data

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    args = argparse.ArgumentParser()
    args.add_argument("--maindir", type=str)
    args.add_argument("--filedir", type=str)
    args.add_argument("--tissues", type=str, nargs="+")
    args.add_argument("--dropbox_dest_path", type=str)
    args = args.parse_args()

    # Get access token from environment variables
    ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')
    if not ACCESS_TOKEN:
        raise ValueError("DROPBOX_ACCESS_TOKEN environment variable not found!\n"
                         "Please set DROPBOX_ACCESS_TOKEN=your_token_here in .env file")

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

    upload_results = []

    # Load atlas and query results
    with open(FILEDIR / "results/atlas_result.json") as f:
        atlas_result = json.load(f)
    with open(FILEDIR / "results/query_result.json") as f:
        query_result = json.load(f)

    for tissue in tissues:
        # Process atlas datasets
        large_dataset_ids = atlas_result[tissue][0]
        small_dataset_ids = atlas_result[tissue][1]

        # Upload large atlas datasets
        for dataset_id in large_dataset_ids:
            data, local_path = get_data(dataset_id=dataset_id, in_atlas=True, large=True)
            upload_results.append(
                get_ans(dataset_id=dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))

        # Upload small atlas datasets
        for dataset_id in small_dataset_ids:
            data, local_path = get_data(dataset_id=dataset_id, in_atlas=True, large=False)
            upload_results.append(
                get_ans(dataset_id=dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))

        # Process query datasets
        large_query_ids = query_result[tissue][0]
        small_query_ids = query_result[tissue][1]

        # Upload large query datasets
        for dataset_id in large_query_ids:
            data, local_path = get_data(dataset_id=dataset_id, in_atlas=False, large=True)
            upload_results.append(
                get_ans(dataset_id=dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))

        # Upload small query datasets
        for dataset_id in small_query_ids:
            data, local_path = get_data(dataset_id=dataset_id, in_atlas=False, large=False)
            upload_results.append(
                get_ans(dataset_id=dataset_id, tissue=tissue, data=data, local_path=local_path,
                        ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH))

    # Save upload results
    output_filename = f"{','.join(tissues)}_scdeepsort.csv"
    pd.DataFrame(upload_results).set_index("species").to_csv(output_filename)
