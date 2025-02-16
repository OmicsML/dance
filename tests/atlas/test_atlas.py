"""Test suite for the Atlas similarity calculation functionality.

This test verifies that the main function correctly returns:
1. The most similar dataset from the atlas
2. Its corresponding configuration settings
3. The similarity score

"""

import json
import sys

import pandas as pd
import pytest

from dance.settings import ATLASDIR, DANCEDIR, SIMILARITYDIR

sys.path.append(str(ATLASDIR))
from demos.main import main

from dance import logger


@pytest.mark.skip(reason="Skipping test due to sensitive data")
def test_main():
    # Construct test parameters with a sample Brain tissue dataset
    class Args:
        tissue = "Brain"
        data_dir = str(DANCEDIR / "tests/atlas/data")
        source_file = "human_Brain576f193c-75d0-4a11-bd25-8676587e6dc2_data"

    args = Args()
    logger.info(f"testing main with args: {args}")
    source_id = "576f"

    # Execute main function with test parameters
    ans_file, ans_conf, ans_value = main(args)

    # Verify return value types and ranges
    assert isinstance(ans_file, str), "ans_file should be a string type"
    assert isinstance(ans_value, float), "ans_value should be a float type"
    assert 0 <= ans_value <= 1, "Similarity value should be between 0 and 1"

    # Verify configuration dictionary structure and content
    expected_methods = ["cta_celltypist", "cta_scdeepsort", "cta_singlecellnet", "cta_actinn"]
    assert isinstance(ans_conf, dict), "ans_conf should be a dictionary type"
    assert set(ans_conf.keys()) == set(expected_methods), "ans_conf should contain all expected methods"
    assert all(isinstance(v, str) for v in ans_conf.values()), "All configuration values should be string type"

    # Verify consistency with Excel spreadsheet results
    data = pd.read_excel(SIMILARITYDIR / f"data/new_sim/{args.tissue.lower()}_similarity.xlsx", sheet_name=source_id,
                         index_col=0)
    reduce_error = False
    in_query = True
    # Read weights
    with open(
            SIMILARITYDIR /
            f"data/similarity_weights_results/{'reduce_error_' if reduce_error else ''}{'in_query_' if in_query else ''}sim_dict.json",
            encoding='utf-8') as f:
        sim_dict = json.load(f)
        feature_name = sim_dict[args.tissue.lower()]["feature_name"]
        w1 = sim_dict[args.tissue.lower()]["weight1"]
        w2 = 1 - w1

    # Calculate similarity in Excel
    data.loc["similarity"] = data.loc[feature_name] * w1 + data.loc["metadata_sim"] * w2
    expected_file = data.loc["similarity"].idxmax()
    expected_value = data.loc["similarity", expected_file]

    # Verify result consistency with Excel
    assert abs(ans_value - expected_value) < 1e-4, "Calculated similarity value does not match Excel value"
    assert ans_file == expected_file, "Selected most similar dataset does not match Excel result"
