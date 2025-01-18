import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from dance.settings import DANCEDIR

sys.path.append(str(DANCEDIR))
from examples.atlas.get_result_web import check_exist, check_identical_strings, spilt_web, write_ans


# 测试 check_identical_strings 函数
def test_check_identical_strings():
    # 测试相同字符串的情况
    assert check_identical_strings(["test", "test", "test"]) == "test"

    # 测试空列表
    with pytest.raises(ValueError, match="The list is empty"):
        check_identical_strings([])

    # 测试不同字符串
    with pytest.raises(ValueError, match="Different strings found"):
        check_identical_strings(["test1", "test2"])


# 测试 spilt_web 函数
def test_spilt_web():
    # 测试有效的URL
    url = "https://wandb.ai/user123/project456/sweeps/abc789"
    result = spilt_web(url)
    assert result == ("user123", "project456", "abc789")

    # 测试无效的URL
    invalid_url = "https://invalid-url.com"
    assert spilt_web(invalid_url) is None


# 测试 check_exist 函数
def test_check_exist(tmp_path):
    # 创建临时测试目录
    results_dir = tmp_path / "results" / "params"
    results_dir.mkdir(parents=True)

    # 测试空目录
    assert check_exist(str(tmp_path)) is False

    # 创建测试文件
    (results_dir / "file1.txt").touch()
    (results_dir / "file2.txt").touch()

    # 测试有多个文件的情况
    assert check_exist(str(tmp_path)) is True


# 创建测试固定装置
@pytest.fixture
def sample_df():
    return pd.DataFrame({"id": ["run1", "run2", "run3"], "metric": [0.8, 0.9, 0.7]})


# 如果需要模拟wandb API，可以使用mock
@pytest.fixture
def mock_wandb(mocker):
    mock_api = mocker.patch("wandb.Api")
    # 这里可以设置mock的返回值
    return mock_api


# 添加一个mock fixture来模拟ATLASDIR
@pytest.fixture(autouse=True)
def mock_settings(tmp_path, monkeypatch):
    """Mock ATLASDIR and METADIR settings for tests."""
    # 创建临时目录
    mock_atlas_dir = tmp_path / "atlas"
    mock_meta_dir = tmp_path / "meta"
    mock_atlas_dir.mkdir(parents=True)
    mock_meta_dir.mkdir(parents=True)

    # 设置环境变量
    monkeypatch.setenv("ATLAS_DIR", str(mock_atlas_dir))
    monkeypatch.setenv("META_DIR", str(mock_meta_dir))

    # 如果直接从dance.settings导入，也替换这些值
    monkeypatch.setattr("examples.atlas.get_result_web.ATLASDIR", mock_atlas_dir)
    monkeypatch.setattr("examples.atlas.get_result_web.METADIR", mock_meta_dir)

    return mock_atlas_dir


def test_write_ans(mock_settings):
    sweep_results_dir = mock_settings / "sweep_results"
    sweep_results_dir.mkdir(parents=True)
    output_file = sweep_results_dir / "heart_ans.csv"

    # 创建初始数据
    existing_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2'],
        'cta_actinn': ['url1', 'url2'],
        'cta_actinn_best_yaml': ['yaml1', 'yaml2'],
        'cta_actinn_best_res': [0.8, 0.7]
    })
    existing_data.to_csv(output_file)

    # 测试数据：包含较低分数和较高分数的情况
    new_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2'],
        'cta_actinn': ['url1_new', 'url2_new'],
        'cta_actinn_best_yaml': ['yaml1_new', 'yaml2_new'],
        'cta_actinn_best_res': [0.9, 0.6]  # dataset1更高分数，dataset2更低分数
    })

    write_ans("heart", new_data, output_file)

    # 验证结果
    result_df = pd.read_csv(output_file)

    # 验证高分数更新成功
    dataset1_row = result_df[result_df['Dataset_id'] == 'dataset1'].iloc[0]
    assert dataset1_row['cta_actinn_best_res'] == 0.9
    assert dataset1_row['cta_actinn'] == 'url1_new'
    assert dataset1_row['cta_actinn_best_yaml'] == 'yaml1_new'

    # 验证低分数保持不变
    dataset2_row = result_df[result_df['Dataset_id'] == 'dataset2'].iloc[0]
    assert dataset2_row['cta_actinn_best_res'] == 0.7
    assert dataset2_row['cta_actinn'] == 'url2'
    assert dataset2_row['cta_actinn_best_yaml'] == 'yaml2'


# 测试完全新的数据写入（文件不存在的情况）
def test_write_ans_new_file(mock_settings):
    # 使用mock_settings而不是创建新的临时目录
    sweep_results_dir = mock_settings / "sweep_results"
    sweep_results_dir.mkdir(parents=True)
    output_file = sweep_results_dir / "new_heart_ans.csv"

    new_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2'],
        'cta_actinn': ['url1', 'url2'],
        'cta_actinn_best_yaml': ['yaml1', 'yaml2'],
        'cta_actinn_best_res': [0.8, 0.9]
    })

    # 测试写入新文件

    # 验证文件被创建并包含正确的数据
    write_ans("heart", new_data, output_file)
    assert output_file.exists()

    written_df = pd.read_csv(output_file, index_col='Dataset_id')
    assert len(written_df) == 2
    assert all(written_df.index == ['dataset1', 'dataset2'])
