import os
from pathlib import Path

import pandas as pd
import pytest
from get_result_web import check_exist, check_identical_strings, spilt_web, write_ans


def test_check_identical_strings():
    # 测试相同字符串
    assert check_identical_strings(['test', 'test', 'test']) == 'test'

    # 测试空列表
    with pytest.raises(ValueError, match="The list is empty"):
        check_identical_strings([])

    # 测试不同字符串
    with pytest.raises(ValueError, match="Different strings found"):
        check_identical_strings(['test1', 'test2'])


def test_spilt_web():
    # 测试正常URL
    url = "https://wandb.ai/xzy11632/dance-dev/sweeps/abc123"
    assert spilt_web(url) == ("xzy11632", "dance-dev", "abc123")

    # 测试无效URL
    assert spilt_web("invalid_url") is None


def test_check_exist(tmp_path):
    # 创建测试目录结构
    results_dir = tmp_path / "results" / "params"
    results_dir.mkdir(parents=True)

    # 测试空目录
    assert check_exist(str(tmp_path)) is False

    # 创建测试文件
    (results_dir / "file1.txt").touch()
    (results_dir / "file2.txt").touch()
    assert check_exist(str(tmp_path)) is True


def test_write_ans(tmp_path):
    """测试write_ans函数的数据更新和冲突检测功能."""
    # 创建测试目录
    output_dir = tmp_path / "atlas" / "sweep_results"
    output_dir.mkdir(parents=True)

    tissue = "heart"
    output_file = output_dir / f"{tissue}_ans.csv"

    # 创建测试数据
    existing_data = {
        'Dataset_id': ['dataset1', 'dataset2'],
        'method1': ['url1', 'url2'],
        'method1_best_yaml': ['yaml1', 'yaml2'],
        'method1_best_res': [0.8, 0.9]
    }
    existing_df = pd.DataFrame(existing_data)
    existing_df.to_csv(output_file)

    # 创建新数据
    new_data = {
        'Dataset_id': ['dataset2', 'dataset3'],
        'method1': ['url2_new', 'url3'],
        'method1_best_yaml': ['yaml2_new', 'yaml3'],
        'method1_best_res': [0.9, 0.95]  # dataset2的结果相同，不会引发冲突
    }
    new_df = pd.DataFrame(new_data)

    # 测试正常更新
    write_ans(tissue, new_df, output_file=output_file)

    # 读取更新后的文件
    updated_df = pd.read_csv(output_file, index_col=0)
    print(updated_df)
    # 验证结果
    assert len(updated_df) == 3  # 应该有3个不同的数据集
    assert 'dataset3' in updated_df.index.values
    assert updated_df.loc['dataset2', 'method1_best_res'] == 0.9

    # 测试结果冲突
    conflict_data = {
        'Dataset_id': ['dataset1'],
        'method1': ['url1_new'],
        'method1_best_yaml': ['yaml1_new'],
        'method1_best_res': [0.7]  # 与现有的0.8不同，应该引发冲突
    }
    conflict_df = pd.DataFrame(conflict_data)

    # 验证冲突检测
    with pytest.raises(ValueError, match="结果冲突"):
        write_ans(tissue, conflict_df, output_file=output_file)


def test_write_ans_new_file(tmp_path):
    """测试write_ans函数创建新文件的功能."""
    # 创建测试目录
    output_dir = tmp_path / "atlas" / "sweep_results"
    output_dir.mkdir(parents=True)

    tissue = "heart"

    # 创建新数据
    new_data = {
        'Dataset_id': ['dataset1', 'dataset2'],
        'method1': ['url1', 'url2'],
        'method1_best_yaml': ['yaml1', 'yaml2'],
        'method1_best_res': [0.8, 0.9]
    }
    new_df = pd.DataFrame(new_data)

    # 测试创建新文件

    # 验证文件创建和内容
    output_file = output_dir / f"{tissue}_ans.csv"
    write_ans(tissue, new_df, output_file=output_file)
    assert output_file.exists()

    # 读取并验证内容
    saved_df = pd.read_csv(output_file, index_col=0)
    assert len(saved_df) == 2
    assert all(saved_df['method1_best_res'] == [0.8, 0.9])
