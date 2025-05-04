# This script finds the most similar dataset from the atlas for a given user-uploaded dataset
# It calculates similarity scores and returns the best matching dataset along with its configurations

import argparse
import json

import pandas as pd
import scanpy as sc

from dance import logger
from dance.atlas.sc_similarity.anndata_similarity import AnnDataSimilarity, get_anndata
from dance.settings import DANCEDIR, SIMILARITYDIR


def calculate_similarity(source_data, tissue, atlas_datasets, reduce_error, in_query):
    """Calculate similarity scores between source data and atlas datasets.

    Args:
        source_data: User uploaded AnnData object
        tissue: Target tissue type
        atlas_datasets: List of candidate datasets from atlas
        reduce_error: Flag for error reduction mode - when True, applies a significant penalty
                     to configurations in the atlas that produced errors
        in_query: Flag for query mode - when True, ranks similarity based on query performance,
                 when False, ranks based on inter-atlas comparison

    Returns:
        Dictionary containing similarity scores for each atlas dataset

    """
    with open(
            SIMILARITYDIR /
            f"data/similarity_weights_results/{'reduce_error_' if reduce_error else ''}{'in_query_' if in_query else ''}sim_dict.json",
            encoding='utf-8') as f:
        sim_dict = json.load(f)
    feature_name = sim_dict[tissue]["feature_name"]
    w1 = sim_dict[tissue]["weight1"]
    w2 = 1 - w1
    ans = {}
    for target_file in atlas_datasets:
        logger.info(f"calculating similarity for {target_file}")
        atlas_data = get_anndata(tissue=tissue.capitalize(), species="human", filetype="h5ad",
                                 train_dataset=[f"{target_file}"], data_dir=str(DANCEDIR / "examples/tuning/temp_data"))
        similarity_calculator = AnnDataSimilarity(adata1=source_data, adata2=atlas_data, sample_size=10,
                                                  init_random_state=42, n_runs=1, tissue=tissue)
        sim_target = similarity_calculator.get_similarity_matrix_A2B(methods=[feature_name, "metadata_sim"])
        ans[target_file] = sim_target[feature_name] * w1 + sim_target["metadata_sim"] * w2
    return ans


def main(args):
    """Main function to process user data and find the most similar atlas dataset.

    Args:
        args: Arguments containing:
            - tissue: Target tissue type
            - data_dir: Directory containing the source data
            - source_file: Name of the source file

    Returns:
        tuple containing:
        - ans_file: ID of the most similar dataset
        - ans_conf: Preprocess configuration dictionary for different cell type annotation methods
        - ans_value: Similarity score of the best matching dataset

    """
    reduce_error = False
    in_query = True
    tissue = args.tissue
    tissue = tissue.lower()
    conf_data = pd.read_excel(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", sheet_name=tissue)
    atlas_datasets = list(conf_data[conf_data["queryed"] == False]["dataset_id"])
    source_data = sc.read_h5ad(f"{args.data_dir}/{args.source_file}.h5ad")

    ans = calculate_similarity(source_data, tissue, atlas_datasets, reduce_error, in_query)
    ans_file = max(ans, key=ans.get)
    ans_value = ans[ans_file]
    ans_conf = {
        method: conf_data.loc[conf_data["dataset_id"] == ans_file, f"{method}_step2_best_yaml"].iloc[0]
        for method in ["cta_celltypist", "cta_scdeepsort", "cta_singlecellnet", "cta_actinn"]
    }
    return ans_file, ans_conf, ans_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", default="Brain")
    parser.add_argument("--data_dir", default=str(DANCEDIR / "examples/tuning/temp_data/train/human"))
    parser.add_argument("--source_file", default="human_Brain364348b4-bc34-4fe1-a851-60d99e36cafa_data")

    args = parser.parse_args()
    ans_file, ans_conf, ans_value = main(args)
    print(ans_file, ans_conf, ans_value)
