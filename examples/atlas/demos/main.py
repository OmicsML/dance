import base64
import io
import json
import os
import sys
import tempfile
import uuid
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from matplotlib import pyplot as plt

from dance import logger
from dance.atlas.sc_similarity.anndata_similarity import AnnDataSimilarity, get_anndata
from dance.pipeline import get_additional_sweep
from dance.settings import DANCEDIR, SIMILARITYDIR, entity, project
from dance.utils import try_import

# --- FastAPI related imports ---

sys.path.append(str(DANCEDIR))
from examples.atlas.sc_similarity_examples.similarity.analyze_atlas_accuracy import is_matching_dict
from examples.atlas.sc_similarity_examples.similarity.process_tissue_similarity_matrices import (
    convert_to_complex,
    unify_complex_float_types_cell,
)
from examples.atlas.sc_similarity_examples.visualization.vis_sim_v2_data import exclude_data, get_atlas_ans
from examples.atlas.sc_similarity_examples.visualization.vis_sim_v2_vis import plot_pre_normalized_radar_v3
from examples.atlas.sc_similarity_examples.visualization.visualize_atlas_performance_v2 import plot_combined_methods

# feature_names_global = ["wasserstein", "Hausdorff", "chamfer", "energy", "sinkhorn2", "bures", "spectral", "mmd","metadata_sim"]
feature_names_global = ["wasserstein", "Hausdorff", "spectral"]

wandb = try_import("wandb")
data_dir = DANCEDIR / f"examples/tuning/temp_data"


# Helper function: convert Matplotlib figure object to Base64 string
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_bytes = buf.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    plt.close(fig)  # Important: close figure to prevent memory leaks
    return base64_string


def get_sim(adata: ad.AnnData, tissue: str, sweep_dict: Optional[dict] = None, feature_name: str = "bures",
            use_sim_cache=False, query_dataset=None):
    """Analyze similarity between user query dataset and atlas datasets, returning the
    most similar dataset and its best preprocessing workflows.

    The main purpose of this function is: given a user's query dataset, the function returns the most similar dataset in the atlas and its best preprocessing workflows across different methods.

    Parameters:
    -----------
    adata : ad.AnnData
        User's query AnnData object
    tissue : str
        Tissue type of the query dataset (e.g., 'brain', 'heart', etc.)
    sweep_dict : Optional[dict], default=None
        Dictionary containing sweep configurations that have been run on the query dataset, format:
        {"cta_actinn": "sweep_id1", "cta_celltypist": "sweep_id2", ...}
        Usually not provided in most cases, in which case plot2 will not be generated
    feature_name : str, default="bures"
        Feature name used to evaluate similarity between query dataset and atlas datasets
    use_sim_cache : bool, default=False
        Whether to use cached similarity matrix, can improve computation speed
    query_dataset : Optional[str], default=None
        Cache name, used when use_sim_cache=True

    Returns:
    --------
    dict
        Dictionary containing the following key-value pairs:
        - metadata: Best preprocessing workflow configurations for the most similar dataset in atlas across different methods
        - plot1_png_base64: Base64 encoded plot1 (radar chart showing similarity scores of the most similar dataset using other functions except feature_name)
        - plot2_png_base64: Base64 encoded plot2 (showing position of found preprocessing workflows in previously run searches, None when sweep_dict=None)

    Notes:
    ------
    - Plot1 displays the performance of the most similar dataset across multiple similarity metrics (in radar chart format)
    - Plot2 is only generated when sweep_dict is provided, showing the position of best preprocessing workflows in historical search results
    - Supported similarity features include: wasserstein, Hausdorff, spectral, etc.
    - Supported methods include: cta_actinn, cta_celltypist, cta_scdeepsort, cta_singlecellnet

    """
    conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
    atlas_datasets = list(conf_data[conf_data["queryed"] == False]["dataset_id"])
    ans = {}
    feature_names = feature_names_global.copy()
    df_excel = pd.ExcelFile(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx")
    if use_sim_cache and query_dataset is not None and query_dataset[:4] in df_excel.sheet_names:
        sim_data = pd.read_excel(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx", sheet_name=query_dataset[:4],
                                 index_col=0)
        for target_file in atlas_datasets:
            ans[target_file] = dict(sim_data.loc[feature_names, target_file])
    else:
        for target_file in atlas_datasets:
            # source_data=sc.read_h5ad(f"{data_root}/{source_file}.h5ad")
            # target_data=sc.read_h5ad(f"{data_root}/{target_file}.h5ad")
            target_data = get_anndata(train_dataset=[f"{target_file}"], data_dir=data_dir, tissue=tissue.capitalize())

            # Initialize similarity calculator with multiple metrics
            similarity_calculator = AnnDataSimilarity(
                adata1=adata, adata2=target_data, sample_size=10, init_random_state=42, n_runs=1,
                ground_truth_conf_path=SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", tissue=tissue)

            # Calculate similarity using multiple methods
            ans[target_file] = similarity_calculator.get_similarity_matrix_A2B(methods=feature_names)
    df = pd.DataFrame(ans)
    df = df[~df.index.duplicated(keep='last')]
    # df=unify_complex_float_types_row(df) #Some complex numbers may lose precision, but it's not a big issue since only real parts are used for comparison
    df = unify_complex_float_types_cell(
        df
    )  #Some complex numbers may lose precision, but it's not a big issue since only real parts are used for comparison
    df.drop(exclude_data.get(tissue, []), axis=1, inplace=True, errors='ignore')
    methods = ["cta_actinn", "cta_celltypist", "cta_scdeepsort", "cta_singlecellnet"]
    df.loc[feature_name, :] = df.loc[feature_name, :].apply(convert_to_complex)
    weighted_sum = (df.loc[feature_name, :]).astype(float)
    atlas_dataset_res = weighted_sum.idxmax()  # Get most similar dataset
    # for i, method_key in enumerate(methods):  # Renamed 'method' to 'method_key'
    # target_value_str, current_atlas_dataset = get_atlas_ans(None, method_key,feature_name=feature_name,data=df) # only for current_atlas_dataset
    ans_conf = {
        method: conf_data.loc[conf_data["dataset_id"] == atlas_dataset_res, f"{method}_step2_best_yaml"].iloc[0]
        for method in ["cta_celltypist", "cta_scdeepsort", "cta_singlecellnet", "cta_actinn"]
    }
    ans_conf["dataset_id"] = atlas_dataset_res
    if sweep_dict is not None:
        method_accs_cache = {}

        for method in methods:
            sweep_id = sweep_dict.get(method, {})
            sweep_ids = get_additional_sweep(entity=entity, project=project, sweep_id=sweep_id)
            accs = []
            runs = []
            for sweep_id in sweep_ids:
                runs.extend(wandb.Api().sweep(f"{entity}/{project}/{sweep_id}").runs)
            accs = [run.summary.get("test_acc", 0) for run in runs]
            method_accs_cache[method] = accs
            for atlas_dataset in atlas_datasets:
                best_yaml = conf_data[conf_data["dataset_id"] == atlas_dataset][f"{method}_step2_best_yaml"].iloc[0]
                match_run = None
                # Find matching run configuration
                for run in runs:
                    if isinstance(best_yaml, float) and np.isnan(best_yaml):
                        continue
                    if is_matching_dict(best_yaml, run.config):
                        if match_run is not None:
                            raise ValueError("Multiple matching runs found when only one expected")
                        match_run = run

                if match_run is None:
                    logger.warning(f"No matching configuration found for {atlas_dataset} with method {method}")
                else:
                    df.loc[method, f"{atlas_dataset}"] = (match_run.summary["test_acc"]
                                                          if "test_acc" in match_run.summary else np.nan)
        if "average_acc" not in df.index:
            df.loc["average_acc"] = df.loc[methods, :].fillna(0).mean(axis=0)
        if "average_acc" not in feature_names:
            feature_names.append("average_acc")

    df_sim = df.loc[feature_names, :].T.applymap(convert_to_complex)
    fig1, _ = plot_pre_normalized_radar_v3(df_sim, atlas_dataset_res, tissue=tissue, query_dataset=None,
                                           title_fontsize=14, other_fill=False)
    b64_image1 = fig_to_base64(fig1)

    b64_image2 = None
    if sweep_dict is not None:
        fig2, _ = plot_combined_methods(df, tissue=tissue, query_dataset=None, methods=methods,
                                        feature_name=feature_name, conf_data=conf_data, save=False,
                                        method_runs_cache=method_accs_cache)
        b64_image2 = fig_to_base64(fig2)
    # 4. Package all content into a Python dictionary
    response_data = {"metadata": ans_conf, "plot1_png_base64": b64_image1, "plot2_png_base64": b64_image2}

    # FastAPI will automatically convert dictionary to JSON response
    return response_data


# ----------------- New FastAPI section -----------------
app = FastAPI()


@app.get("/api/get_method")
async def get_atlas_method(atlas_id, tissue):
    """Retrieve the best preprocessing workflows for a specific atlas dataset across
    different methods.

    This endpoint returns the optimal preprocessing configurations for a given atlas dataset
    identified by its ID and tissue type, across all supported cell type annotation methods.

    Parameters:
    -----------
    atlas_id : str
        The ID of the atlas dataset to retrieve preprocessing workflows for
    tissue : str
        The tissue type of the atlas dataset (e.g., 'brain', 'heart', etc.)

    Returns:
    --------
    dict
        Dictionary containing the best preprocessing workflow configurations for the specified
        atlas dataset across different methods:
        - cta_celltypist: Best preprocessing workflow for CellTypist method
        - cta_scdeepsort: Best preprocessing workflow for scDeepSort method
        - cta_singlecellnet: Best preprocessing workflow for SingleCellNet method
        - cta_actinn: Best preprocessing workflow for ACTINN method
        - dataset_id: The atlas dataset ID

    Notes:
    ------
    - The workflows are retrieved from the Cell Type Annotation Atlas Excel file
    - Supported methods include: cta_celltypist, cta_scdeepsort, cta_singlecellnet, cta_actinn
    - Each method returns its optimal preprocessing configuration in YAML format

    """

    conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
    ans_conf = {
        method: conf_data.loc[conf_data["dataset_id"] == atlas_id, f"{method}_step2_best_yaml"].iloc[0]
        for method in ["cta_celltypist", "cta_scdeepsort", "cta_singlecellnet", "cta_actinn"]
    }
    ans_conf["dataset_id"] = atlas_id
    return ans_conf


@app.post("/api/get_similarity")
async def run_similarity_analysis(h5ad_file: UploadFile = File(..., description="Upload .h5ad format query data file"),
                                  tissue: str = Form(..., description="Tissue type, e.g. 'brain'"),
                                  feature_name: str = Form("metadata_sim", description="Feature name to use"),
                                  use_sim_cache: bool = Form(False,
                                                             description="Whether to use cached similarity matrix"),
                                  query_dataset: Optional[str] = Form(None, description="Query dataset ID"),
                                  sweep_dict_json: Optional[str] = Form(
                                      None, description="JSON string containing sweep IDs")):
    """FastAPI wrapper for the get_sim function to perform similarity analysis via HTTP
    API.

    This endpoint receives an uploaded h5ad file and analysis parameters, then calls the get_sim function
    to analyze similarity between the query dataset and atlas datasets. It returns the most similar
    dataset from the atlas along with its best preprocessing workflows and visualization plots.

    Parameters:
    -----------
    h5ad_file : UploadFile
        Uploaded .h5ad format query data file
    tissue : str
        Tissue type of the query dataset (e.g., 'brain', 'heart', etc.)
    feature_name : str, default="metadata_sim"
        Feature name used to evaluate similarity between query and atlas datasets
    use_sim_cache : bool, default=False
        Whether to use cached similarity matrix for faster computation
    query_dataset : Optional[str], default=None
        Query dataset ID for cache identification
    sweep_dict_json : Optional[str], default=None
        JSON string containing sweep configurations that have been run on the query dataset

    Returns:
    --------
    dict
        Same return format as get_sim function:
        - metadata: Best preprocessing workflow configurations for the most similar atlas dataset
        - plot1_png_base64: Base64 encoded radar chart showing similarity metrics
        - plot2_png_base64: Base64 encoded plot showing preprocessing workflow positions (if sweep_dict provided)

    Notes:
    ------
    - This is essentially a FastAPI wrapper around the get_sim function
    - The function handles file upload, parameter processing, and temporary file cleanup
    - All core analysis logic is delegated to the get_sim function
    - Returns the same analysis results as get_sim but through HTTP API interface

    """
    # 1. Process uploaded file
    # Create a secure temporary file to save uploaded content
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.h5ad")

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await h5ad_file.read())

        # Use scanpy to read temporary file
        adata = sc.read_h5ad(temp_file_path)

        # 2. Process sweep_dict
        sweep_dict = None
        if sweep_dict_json:
            try:
                sweep_dict = json.loads(sweep_dict_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="sweep_dict_json is not a valid JSON string.")

        # 3. Call your core analysis function
        logger.info(f"Starting analysis tissue={tissue}, feature_name={feature_name}...")
        results = get_sim(adata=adata, tissue=tissue, sweep_dict=sweep_dict, feature_name=feature_name,
                          use_sim_cache=use_sim_cache, query_dataset=query_dataset)
        logger.info("Analysis completed.")

        return results

    except Exception as e:
        # Catch all possible errors and return a meaningful error message
        logger.error(f"Error occurred during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # 4. Clean up temporary files, whether successful or failed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

# Command to start the service (run in terminal)
# uvicorn main:app --host 0.0.0.0 --port 8100 --reload

# if __name__ == "__main__":
#     adata=sc.read_h5ad("/home/zyxing/dance/examples/tuning/temp_data/train/human/human_Brain576f193c-75d0-4a11-bd25-8676587e6dc2_data.h5ad")
#     tissue="brain"
#     sweep_dict={"cta_actinn":"91txflmo",
#                 "cta_celltypist":"l2m0ex0v",
#                 "cta_scdeepsort":"x78ukq8v",
#                 "cta_singlecellnet":"cnzh26nr"}
#     feature_name="metadata_sim"
    ans = get_sim(adata, tissue, sweep_dict, feature_name=feature_name, use_sim_cache=True,
                  query_dataset="576f193c-75d0-4a11-bd25-8676587e6dc2")


#     print("To start the API service, run in terminal: uvicorn main:app --reload")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
