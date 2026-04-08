"""汇总所有 summary_all_results_*.json 文件，生成包含 alpha、weighted_score_mean 和 folder_name 的总
CSV."""
import csv
import json
import os
import re
from pathlib import Path


def extract_weighted_scores_from_json(json_path: str, alpha: float):
    """从 JSON 文件中提取 folder_name 和 weighted_score_mean.

    Args:
        json_path: JSON 文件路径
        alpha: alpha 值

    Returns:
        包含 folder_name 和 weighted_score_mean 的字典列表

    """
    # 读取 JSON 文件
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    # 提取数据
    results = []
    for folder_name, folder_data in data.items():
        best_combo = folder_data.get('best_param_combination', {})
        weighted_score_mean = best_combo.get('weighted_score_mean')

        results.append({'alpha': alpha, 'folder_name': folder_name, 'weighted_score_mean': weighted_score_mean})

    return results


def summarize_all_json_files(script_dir: Path, output_csv: str = None):
    """汇总目录下所有 summary_all_results_*.json 和 summary_all_results.json 文件 不带 alpha 后缀的
    json 文件默认为 alpha=0.8."""
    # 默认输出文件名
    if output_csv is None:
        output_csv = script_dir / 'summary_all_weighted_scores.csv'
    elif not os.path.isabs(output_csv):
        output_csv = script_dir / output_csv

    # 查找所有 summary_all_results_*.json 文件（带 alpha 后缀的）
    json_files_with_alpha = sorted(script_dir.glob('summary_all_results_*.json'))

    # 检查是否存在 summary_all_results.json（不带 alpha 后缀，默认为 alpha=0.8）
    default_json = script_dir / 'summary_all_results.json'
    has_default_json = default_json.exists()

    if not json_files_with_alpha and not has_default_json:
        print("未找到任何 summary_all_results*.json 文件")
        return

    print(f"找到带 alpha 后缀的 JSON 文件: {len(json_files_with_alpha)} 个")
    for f in json_files_with_alpha:
        print(f"  - {f.name}")

    if has_default_json:
        print(f"  - summary_all_results.json (alpha=0.8)")

    # 提取 alpha 值并收集所有数据
    all_results = []

    # 处理带 alpha 后缀的 JSON 文件
    for json_file in json_files_with_alpha:
        # 从文件名提取 alpha 值
        match = re.search(r'alpha(\d+\.\d+)', json_file.name)
        if match:
            alpha = float(match.group(1))
        else:
            alpha = None
            print(f"警告: 无法从 {json_file.name} 提取 alpha 值")

        # 提取数据
        results = extract_weighted_scores_from_json(str(json_file), alpha)
        all_results.extend(results)

        print(f"从 {json_file.name} 提取了 {len(results)} 条记录 (alpha={alpha})")

    # 处理不带 alpha 后缀的 JSON 文件（默认为 alpha=0.8）
    if has_default_json:
        alpha = 0.8
        results = extract_weighted_scores_from_json(str(default_json), alpha)
        all_results.extend(results)
        print(f"从 summary_all_results.json 提取了 {len(results)} 条记录 (alpha={alpha})")

    # 按 alpha 降序、weighted_score_mean 降序排序
    all_results.sort(
        key=lambda x: (x['alpha'] if x['alpha'] else 0, x['weighted_score_mean']
                       if x['weighted_score_mean'] else 0), reverse=True)

    # 保存 CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['alpha', 'folder_name', 'weighted_score_mean'])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n共提取 {len(all_results)} 条记录")
    print(f"结果已保存到: {output_csv}")

    # 显示结果预览
    print("\n结果预览:")
    print(f"{'alpha':<8} {'folder_name':<25} {'weighted_score_mean':>20}")
    print("-" * 54)
    for row in all_results:
        alpha_str = f"{row['alpha']:.1f}" if row['alpha'] else "N/A"
        score_str = f"{row['weighted_score_mean']:.4f}" if row['weighted_score_mean'] else "N/A"
        print(f"{alpha_str:<8} {row['folder_name']:<25} {score_str:>20}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="汇总所有 summary_all_results_*.json 文件")
    parser.add_argument('--output', type=str, default='summary_all_weighted_scores.csv', help='输出的 CSV 文件路径')

    args = parser.parse_args()

    # 使用脚本所在目录
    script_dir = Path(__file__).parent.resolve()

    # 执行汇总
    summarize_all_json_files(script_dir, args.output)
