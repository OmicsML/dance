from pathlib import Path

import pandas as pd
import pytest

from examples.get_result_web import check_exist, check_identical_strings, spilt_web


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


def test_write_ans(tmp_path):
    # 模拟 atlas/sweep_results 目录
    sweep_results_dir = tmp_path / "atlas" / "sweep_results"
    sweep_results_dir.mkdir(parents=True)

    # 创建测试数据
    existing_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2', 'dataset3'],
        'method1': ['url1', 'url2', 'url3'],
        'method1_best_yaml': ['yaml1', 'yaml2', 'yaml3'],
        'method1_best_res': [0.8, 0.9, 0.7]
    })

    new_data = pd.DataFrame({
        'Dataset_id': ['dataset2', 'dataset3', 'dataset4'],  # 部分重叠的数据
        'method1': ['url2_new', 'url3_new', 'url4'],
        'method1_best_yaml': ['yaml2_new', 'yaml3_new', 'yaml4'],
        'method1_best_res': [0.9, 0.7, 0.85]  # dataset2和dataset3的结果与现有数据相同
    })

    # 写入现有数据
    output_file = sweep_results_dir / "heart_ans.csv"
    existing_data.to_csv(output_file)

    # 测试写入新数据
    from examples.get_result_web import write_ans
    write_ans("heart", new_data)

    # 读取合并后的结果
    merged_df = pd.read_csv(output_file, index_col=0)

    # 验证结果
    assert len(merged_df) == 4  # 应该有4个唯一的Dataset_id
    assert 'dataset4' in merged_df.index  # 新数据被添加
    assert merged_df.loc['dataset2', 'method1'] == 'url2_new'  # 更新了已存在的数据

    # 测试结果冲突的情况
    conflicting_data = pd.DataFrame({
        'Dataset_id': ['dataset1'],
        'method1': ['url1_new'],
        'method1_best_yaml': ['yaml1_new'],
        'method1_best_res': [0.95]  # 不同的结果值
    })

    # 验证冲突数据会引发异常
    with pytest.raises(ValueError, match="结果冲突"):
        write_ans("heart", conflicting_data)


# 测试完全新的数据写入（文件不存在的情况）
def test_write_ans_new_file(tmp_path):
    # 模拟 atlas/sweep_results 目录
    sweep_results_dir = tmp_path / "atlas" / "sweep_results"
    sweep_results_dir.mkdir(parents=True)

    new_data = pd.DataFrame({
        'Dataset_id': ['dataset1', 'dataset2'],
        'method1': ['url1', 'url2'],
        'method1_best_yaml': ['yaml1', 'yaml2'],
        'method1_best_res': [0.8, 0.9]
    })

    # 测试写入新文件
    from examples.get_result_web import write_ans
    write_ans("heart", new_data)

    # 验证文件被创建并包含正确的数据
    output_file = sweep_results_dir / "heart_ans.csv"
    assert output_file.exists()

    written_df = pd.read_csv(output_file, index_col=0)
    assert len(written_df) == 2
    assert all(written_df.index == ['dataset1', 'dataset2'])
