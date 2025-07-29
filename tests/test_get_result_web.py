import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from dance.settings import DANCEDIR

sys.path.append(str(DANCEDIR))
from examples.atlas.get_result_web import check_exist, check_identical_strings, spilt_web, write_ans


# Test check_identical_strings function
def test_check_identical_strings():
    # Test case for identical strings
    assert check_identical_strings(["test", "test", "test"]) == "test"

    # Test case for empty list
    with pytest.raises(ValueError, match="The list is empty"):
        check_identical_strings([])

    # Test case for different strings
    with pytest.raises(ValueError, match="Different strings found"):
        check_identical_strings(["test1", "test2"])


# Test spilt_web function
def test_spilt_web():
    # Test valid URL
    url = "https://wandb.ai/user123/project456/sweeps/abc789"
    result = spilt_web(url)
    assert result == ("user123", "project456", "abc789")

    # Test invalid URL
    invalid_url = "https://invalid-url.com"
    assert spilt_web(invalid_url) is None


# Test check_exist function
def test_check_exist(tmp_path):
    # Create temporary test directory
    results_dir = tmp_path / "results" / "params"
    results_dir.mkdir(parents=True)

    # Test empty directory
    assert check_exist(str(tmp_path)) is False

    # Create test files
    (results_dir / "file1.txt").touch()
    (results_dir / "file2.txt").touch()

    # Test case with multiple files
    assert check_exist(str(tmp_path)) is True


# Create test fixed data
@pytest.fixture
def sample_df():
    return pd.DataFrame({"id": ["run1", "run2", "run3"], "metric": [0.8, 0.9, 0.7]})


#  use mock to simulate wandb API
@pytest.fixture
def mock_wandb(mocker):
    mock_api = mocker.patch("wandb.Api")
    # set mock return values
    return mock_api


# Add a mock fixture to simulate ATLASDIR
@pytest.fixture(autouse=True)
def mock_settings(tmp_path, monkeypatch):
    """Mock ATLASDIR and METADIR settings for tests."""
    # Create temporary directory
    mock_atlas_dir = tmp_path / "atlas"
    mock_meta_dir = tmp_path / "meta"
    mock_atlas_dir.mkdir(parents=True)
    mock_meta_dir.mkdir(parents=True)

    # Set environment variables
    monkeypatch.setenv("ATLAS_DIR", str(mock_atlas_dir))
    monkeypatch.setenv("META_DIR", str(mock_meta_dir))

    # If import directly from dance.settings, also replace these values
    monkeypatch.setattr("examples.atlas.get_result_web.ATLASDIR", mock_atlas_dir)
    monkeypatch.setattr("examples.atlas.get_result_web.METADIR", mock_meta_dir)

    return mock_atlas_dir


def test_write_ans(mock_settings):
    sweep_results_dir = mock_settings / "sweep_results"
    sweep_results_dir.mkdir(parents=True)
    output_file = sweep_results_dir / "heart_ans.csv"

    # Create initial data
    existing_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2'],
        'cta_actinn': ['url1', 'url2'],
        'cta_actinn_best_yaml': ['yaml1', 'yaml2'],
        'cta_actinn_best_res': [0.8, 0.7]
    })
    existing_data.to_csv(output_file)

    # Test data: include lower score and higher score
    new_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2'],
        'cta_actinn': ['url1_new', 'url2_new'],
        'cta_actinn_best_yaml': ['yaml1_new', 'yaml2_new'],
        'cta_actinn_best_res': [0.9, 0.6]  # dataset1 higher score, dataset2 lower score
    })

    write_ans("heart", new_data, output_file)

    # Verify results
    result_df = pd.read_csv(output_file)

    # Verify high score update success
    dataset1_row = result_df[result_df['Dataset_id'] == 'dataset1'].iloc[0]
    assert dataset1_row['cta_actinn_best_res'] == 0.9
    assert dataset1_row['cta_actinn'] == 'url1_new'
    assert dataset1_row['cta_actinn_best_yaml'] == 'yaml1_new'

    # Verify low score remains unchanged
    dataset2_row = result_df[result_df['Dataset_id'] == 'dataset2'].iloc[0]
    assert dataset2_row['cta_actinn_best_res'] == 0.7
    assert dataset2_row['cta_actinn'] == 'url2'
    assert dataset2_row['cta_actinn_best_yaml'] == 'yaml2'


# Test completely new data write (file does not exist)
def test_write_ans_new_file(mock_settings):
    # Use mock_settings instead of creating new temporary directory
    sweep_results_dir = mock_settings / "sweep_results"
    sweep_results_dir.mkdir(parents=True)
    output_file = sweep_results_dir / "new_heart_ans.csv"

    new_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2'],
        'cta_actinn': ['url1', 'url2'],
        'cta_actinn_best_yaml': ['yaml1', 'yaml2'],
        'cta_actinn_best_res': [0.8, 0.9]
    })

    # Test write to new file

    # Verify file is created and contains correct data
    write_ans("heart", new_data, output_file)
    assert output_file.exists()

    written_df = pd.read_csv(output_file, index_col='Dataset_id')
    assert len(written_df) == 2
    assert all(written_df.index == ['dataset1', 'dataset2'])
