import os
import re
import pandas as pd
from pathlib import Path
import numpy as np
import argparse  # 1. 导入 argparse 库
import json
alpha=0.9
dataset_names=['151507','151673','151676','human_breast_cancer','pancreatic_cancer',"328_138",
               '1013-1247-598-732-767-768-770-784-845-864_315-340-376-381-390-404-437-490-551-559',
               '1027-1357-1641-517-706-777-850-972_245-332-377-398-405-455-470-492','3043-3777-4029-4115-4362-4657_1729-2125-2184-2724-2743',
               '11407-1519-636-713-9054-9258_1925-205-3323-6509-7572'
               ]
def summarize_best_test_acc(base_dir, metric_name):
    """
    汇总指定文件夹及其子文件夹中的 best_test_acc.csv 文件，并根据参数组合分组计算均值
    所有csv文件都是同一方法在不同数据集上的搜索结果
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"错误: 路径 '{base_path}' 不存在。")
        return pd.DataFrame(), pd.DataFrame()

    all_data = []

    # 只处理第一层子文件夹中的数据集文件夹
    for subfolder in base_path.iterdir():
        if not subfolder.is_dir():
            continue

        dataset_name = subfolder.name

        if dataset_name in dataset_names:
            # 检查是否有 results/pipeline/best_test_acc.csv
            csv_path = subfolder /  'results' / 'pipeline' / 'best_test_acc.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    df['dataset'] = dataset_name
                    # 使用传入的文件夹名作为方法名
                    df['method'] = base_path.name
                    all_data.append(df)
                    print(f"[{base_path.name}/{dataset_name}] 读取成功")
                except Exception as e:
                    print(f"[{base_path.name}/{dataset_name}] 读取出错: {e}")

    if not all_data:
        print(f"在 '{base_dir}' 中未找到有效的 best_test_acc.csv 文件")
        return pd.DataFrame(), pd.DataFrame()

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"共读取了 {len(combined_df)} 行数据，来自 {combined_df['dataset'].nunique()} 个数据集")

    # 识别所有以"params"为前缀的列
    param_columns = [col for col in combined_df.columns if col.startswith('params')]

    if not param_columns:
        print("警告: 未找到以'params'为前缀的列")
        return combined_df, pd.DataFrame()

    print(f"找到 {len(param_columns)} 个参数列: {param_columns}")

    # 检查参数列是否有缺失值
    missing_params = combined_df[param_columns].isnull().any().any()
    
    if missing_params:
        print("参数列存在缺失值，将填充为 'N/A'")
        combined_df[param_columns] = combined_df[param_columns].fillna('N/A')
    combined_df.fillna({
            metric_name: 0,
            'speed_score': 0,
            'combined_score': 0,
            })
    combined_df['weighted_score'] = combined_df[metric_name] * alpha + combined_df['speed_score'] * (1-alpha)
    # 根据参数列分组，计算每个参数组合在所有数据集上的统计信息
    grouped = combined_df.groupby(param_columns).agg({
        metric_name: ['mean', 'std', 'count'],
        'speed_score': ['mean', 'std'],
        'combined_score': ['mean', 'std'],
        'weighted_score': ['mean', 'std']
    }).round(4)

    # 展平多级列索引
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()


    # 按combined_score降序排序，找到内部分数最高的参数组合
    grouped_by_combined = grouped.sort_values('combined_score_mean', ascending=False).reset_index(drop=True)

    print(f"生成 {len(grouped)} 个不同的参数组合")
    print(f"内部分数最高的参数组合的测试集加权得分: {grouped_by_combined.iloc[0]['weighted_score_mean']:.4f} ± {grouped_by_combined.iloc[0]['weighted_score_std']:.4f}")

    return combined_df, grouped_by_combined

if __name__ == "__main__":
    # 2. 设置参数解析器
    parser = argparse.ArgumentParser(description="汇总指定目录下的实验结果")

    # 添加 --dir 参数，默认值为脚本所在目录
    parser.add_argument('--dir', type=str, default=None, help='脚本所在目录，如果不指定则自动检测')

    # 添加 --output 参数，允许自定义输出文件名
    parser.add_argument('--output', type=str, default='summary_all_results.json', help='输出 JSON 的文件名')

    args = parser.parse_args()

    # 获取脚本所在目录
    if args.dir is None:
        script_dir = Path(__file__).parent.resolve()
    else:
        script_dir = Path(args.dir).resolve()

    print(f"脚本所在目录: {script_dir}")

    # 遍历第一层子文件夹
    subfolders = [f for f in script_dir.iterdir() if f.is_dir() and (f.name.startswith('cta') or f.name.startswith('domain'))]

    if not subfolders:
        print("未找到以 'cta' 或 'domain' 开头的子文件夹")
        exit(1)

    print(f"找到 {len(subfolders)} 个待处理的子文件夹: {[f.name for f in subfolders]}")

    # 存储所有结果
    all_results = {}

    for subfolder in subfolders:
        folder_name = subfolder.name
        print(f"\n正在处理文件夹: {folder_name}")

        # 根据文件夹名称确定metric_name
        if folder_name.startswith('domain'):
            metric_name = "ARI"
        elif folder_name.startswith('cta'):
            metric_name = "test_acc"
        else:
            print(f"跳过文件夹 {folder_name}（不符合命名规则）")
            continue


        # 汇总数据
        combined_df, grouped_stats_df = summarize_best_test_acc(subfolder, metric_name)

        if not combined_df.empty and not grouped_stats_df.empty:
            print(f"文件夹 {folder_name} 汇总了 {len(combined_df)} 行原始数据")

            # 保存CSV文件
            if alpha==0.8:
                csv_output_path = subfolder / 'summary_results.csv'
            else:
                csv_output_path = subfolder / f'summary_results_alpha{alpha}.csv'
            grouped_stats_df.to_csv(csv_output_path, index=False)
            print(f"CSV结果已保存到: {csv_output_path}")

            # 准备JSON结果
            best_params = grouped_stats_df.iloc[0]

            # 获取参数详情，确保值是JSON可序列化的
            param_columns = [col for col in grouped_stats_df.columns if col.startswith('params')]
            param_details = {}
            for param_col in param_columns:
                param_value = best_params[param_col]
                if param_value != 'N/A':  # 只包含非N/A的参数
                    # 转换为Python原生类型
                    if isinstance(param_value, (int, float)):
                        param_details[param_col] = param_value
                    else:
                        param_details[param_col] = str(param_value)

            # 构建结果字典，确保所有数值都是Python原生类型
            folder_result = {
                "folder_name": folder_name,
                "metric_name": metric_name,
                "total_raw_data_rows": int(len(combined_df)),
                "total_param_combinations": int(len(grouped_stats_df)),
                "best_param_combination": {
                    "combined_score_mean": float(best_params['combined_score_mean']),
                    "combined_score_std": float(best_params['combined_score_std']),
                    "weighted_score_mean": float(best_params['weighted_score_mean']),
                    "weighted_score_std": float(best_params['weighted_score_std']),
                    f"{metric_name}_mean": float(best_params[f'{metric_name}_mean']),
                    f"{metric_name}_std": float(best_params[f'{metric_name}_std']),
                    "speed_score_mean": float(best_params['speed_score_mean']),
                    "speed_score_std": float(best_params['speed_score_std']),
                    "dataset_count": int(best_params[f'{metric_name}_count']),
                    "parameters": param_details
                }
            }

            all_results[folder_name] = folder_result

            # 显示结果
            print(f"平均内部分数 (combined_score): {best_params['combined_score_mean']:.4f} ± {best_params['combined_score_std']:.4f}")
            print(f"测试集加权得分 ({metric_name}*0.8 + speed_score*0.2): {best_params['weighted_score_mean']:.4f} ± {best_params['weighted_score_std']:.4f}")
            print(f"平均{metric_name}: {best_params[f'{metric_name}_mean']:.4f} ± {best_params[f'{metric_name}_std']:.4f}")
            print(f"平均速度得分: {best_params['speed_score_mean']:.4f} ± {best_params['speed_score_std']:.4f}")
            print(f"覆盖数据集数量: {int(best_params[f'{metric_name}_count'])}")

        else:
            print(f"文件夹 {folder_name} 未生成有效数据")

    # 保存大JSON文件
    if all_results:
        json_output_path = script_dir / args.output
        if alpha!=0.8:
            json_output_path = str(json_output_path).replace('.json', f'_alpha{alpha}.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n所有结果已保存到 JSON 文件: {json_output_path}")
        print(f"共处理了 {len(all_results)} 个文件夹")
    else:
        print("未生成任何有效结果")