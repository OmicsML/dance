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
    existing_data = [{
        'Dataset_id': 'dataset1',
        'cta_actinn': 'url1',
        'cta_actinn_best_yaml': 'yaml1',
        'cta_actinn_best_res': 0.8,
        'cta_actinn_run_stats': 'stats1',
        'cta_actinn_check': False
    }, {
        'Dataset_id': 'dataset2',
        'cta_actinn': 'url2',
        'cta_actinn_best_yaml': 'yaml2',
        'cta_actinn_best_res': 0.9,
        'cta_actinn_run_stats': 'stats2',
        'cta_actinn_check': False
    }]
    existing_df = pd.DataFrame(existing_data)
    existing_df.to_csv(output_file)

    # 创建新数据
    new_data = [{
        'Dataset_id': 'dataset2',
        'cta_actinn': 'url2',
        'cta_actinn_best_yaml': 'yaml2',
        'cta_actinn_best_res': 0.85,  # 低于现有值，不应更新
        'cta_actinn_run_stats': 'stats2_new',
        'cta_actinn_check': True
    }, {
        'Dataset_id': 'dataset3',
        'cta_actinn': 'url3',
        'cta_actinn_best_yaml': 'yaml3',
        'cta_actinn_best_res': 0.95,
        'cta_actinn_run_stats': 'stats3',
        'cta_actinn_check': False
    }, {
        'Dataset_id': 'dataset2',
        'cta_celltypist': 'url2_2',
        'cta_celltypist_best_yaml': 'yaml2_2',
        'cta_celltypist_best_res': 0.95,
        'cta_celltypist_run_stats': 'stats2_2',
        'cta_celltypist_check': True
    }]
    new_df = pd.DataFrame(new_data)

    # 测试更新
    write_ans(tissue, new_df, output_file=output_file)
    updated_df = pd.read_csv(output_file)

    # 验证结果
    assert len(updated_df) == 3  # 应该有3个不同的数据集
    assert 'dataset3' in updated_df['Dataset_id'].values
    
    # dataset2的cta_actinn不应该被更新（因为新值较低）
    assert updated_df[updated_df['Dataset_id'] == 'dataset2']['cta_actinn_best_res'].iloc[0] == 0.9
    assert updated_df[updated_df['Dataset_id'] == 'dataset2']['cta_actinn_run_stats'].iloc[0] == 'stats2'
    
    # dataset2的cta_celltypist应该被添加
    assert updated_df[updated_df['Dataset_id'] == 'dataset2']['cta_celltypist_best_res'].iloc[0] == 0.95

    # 测试更高值的更新
    higher_data = [{
        'Dataset_id': 'dataset1',
        'cta_actinn': 'url1_new',
        'cta_actinn_best_yaml': 'yaml1_new',
        'cta_actinn_best_res': 0.9,  # 高于现有值0.8，应该更新
        'cta_actinn_run_stats': 'stats1_new',
        'cta_actinn_check': True
    }]
    higher_df = pd.DataFrame(higher_data)

    write_ans(tissue, higher_df, output_file=output_file)
    final_df = pd.read_csv(output_file)
    
    # 验证所有相关列都被更新
    dataset1_row = final_df[final_df['Dataset_id'] == 'dataset1'].iloc[0]
    assert dataset1_row['cta_actinn_best_res'] == 0.9
    assert dataset1_row['cta_actinn'] == 'url1_new'
    assert dataset1_row['cta_actinn_best_yaml'] == 'yaml1_new'
    assert dataset1_row['cta_actinn_run_stats'] == 'stats1_new'
    assert dataset1_row['cta_actinn_check'] == True


def test_write_ans_new_file(tmp_path):
    """测试write_ans函数创建新文件的功能."""
    # 创建测试目录
    output_dir = tmp_path / "atlas" / "sweep_results"
    output_dir.mkdir(parents=True)

    tissue = "heart"

    # 创建新数据
    new_data = [{
        'Dataset_id': 'dataset1',
        'method1': 'url1',
        'method1_best_yaml': 'yaml1',
        'method1_best_res': 0.8
    }, {
        'Dataset_id': 'dataset2',
        'method1': 'url2',
        'method1_best_yaml': 'yaml2',
        'method1_best_res': 0.9
    }]
    new_df = pd.DataFrame(new_data)

    # 测试创建新文件
    output_file = output_dir / f"{tissue}_ans.csv"
    write_ans(tissue, new_df, output_file=output_file)
    assert output_file.exists()

    # 读取并验证内容
    saved_df = pd.read_csv(output_file, index_col=0)
    assert len(saved_df) == 2
    assert all(saved_df['method1_best_res'] == [0.8, 0.9])
