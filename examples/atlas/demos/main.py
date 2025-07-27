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
from joblib import PrintTime
from matplotlib import pyplot as plt
from networkx import dfs_tree

from dance import logger
from dance.atlas.sc_similarity.anndata_similarity import AnnDataSimilarity, get_anndata
from dance.pipeline import get_additional_sweep
from dance.settings import DANCEDIR, SIMILARITYDIR, entity, project
from dance.utils import try_import

# --- FastAPI 相关的导入 ---

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


# 辅助函数：将Matplotlib figure对象转为Base64字符串
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_bytes = buf.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    plt.close(fig)  # 重要：关闭图形，防止内存泄漏
    return base64_string


def get_sim(adata: ad.AnnData, tissue: str, sweep_dict: Optional[dict] = None, feature_name: str = "bures",
            use_sim_cache=False, query_dataset=None):
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
    # 4. 将所有内容打包到一个Python字典中
    response_data = {"metadata": ans_conf, "plot1_png_base64": b64_image1, "plot2_png_base64": b64_image2}

    # FastAPI会自动将字典转换为JSON响应
    return response_data


# ----------------- 新增 FastAPI 部分 -----------------
app = FastAPI()


@app.get("/api/get_method")
async def get_atlas_method(atlas_id, tissue):
    conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
    ans_conf = {
        method: conf_data.loc[conf_data["dataset_id"] == atlas_id, f"{method}_step2_best_yaml"].iloc[0]
        for method in ["cta_celltypist", "cta_scdeepsort", "cta_singlecellnet", "cta_actinn"]
    }
    ans_conf["dataset_id"] = atlas_id
    return ans_conf


@app.post("/api/get_similarity")
async def run_similarity_analysis(h5ad_file: UploadFile = File(..., description="上传 .h5ad 格式的查询数据文件"),
                                  tissue: str = Form(..., description="组织类型, 例如 'brain'"),
                                  feature_name: str = Form("metadata_sim", description="要使用的特征名称"),
                                  use_sim_cache: bool = Form(False, description="是否使用缓存的相似度矩阵"),
                                  query_dataset: Optional[str] = Form(None, description="查询数据集的ID"),
                                  sweep_dict_json: Optional[str] = Form(None, description="包含sweep ID的JSON字符串")):
    """接收上传的h5ad文件和参数，运行相似度分析，并返回包含结果和图表的JSON。"""
    # 1. 处理上传的文件
    # 创建一个安全的临时文件来保存上传内容
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.h5ad")

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await h5ad_file.read())

        # 使用scanpy读取临时文件
        adata = sc.read_h5ad(temp_file_path)

        # 2. 处理 sweep_dict
        sweep_dict = None
        if sweep_dict_json:
            try:
                sweep_dict = json.loads(sweep_dict_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="sweep_dict_json 不是一个有效的JSON字符串。")

        # 3. 调用你的核心分析函数
        logger.info(f"开始分析 tissue={tissue}, feature_name={feature_name}...")
        results = get_sim(adata=adata, tissue=tissue, sweep_dict=sweep_dict, feature_name=feature_name,
                          use_sim_cache=use_sim_cache, query_dataset=query_dataset)
        logger.info("分析完成。")

        return results

    except Exception as e:
        # 捕获所有可能的错误，并返回一个有意义的错误信息
        logger.error(f"分析过程中发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
    finally:
        # 4. 清理临时文件，无论成功或失败
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"已清理临时文件: {temp_file_path}")

# 启动服务的命令 (在终端中运行)
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


#     print("要启动API服务，请在终端中运行: uvicorn main:app --reload")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
