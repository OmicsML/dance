"""
从 summary_all_results_*.json 文件中提取 weighted_score_mean 和 folder_name
生成一个新的 CSV 文件（使用标准库，无需 pandas）
"""
import os
import json
import csv
from pathlib import Path


def extract_weighted_scores(json_path: str, output_csv: str = None):
    """
    从 JSON 文件中提取 folder_name 和 weighted_score_mean
    
    Args:
        json_path: JSON 文件路径
        output_csv: 输出的 CSV 文件路径（可选，默认与 JSON 同名但扩展名为 .csv）
    """
    # 如果未指定输出路径，则使用 JSON 文件名但扩展名为 .csv
    if output_csv is None:
        output_csv = str(json_path).replace('.json', '_weighted_scores.csv')
    
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取数据
    results = []
    for folder_name, folder_data in data.items():
        best_combo = folder_data.get('best_param_combination', {})
        weighted_score_mean = best_combo.get('weighted_score_mean')
        
        results.append({
            'folder_name': folder_name,
            'weighted_score_mean': weighted_score_mean
        })
    
    # 按 weighted_score_mean 降序排序
    results.sort(key=lambda x: (x['weighted_score_mean'] is not None, 
                                x['weighted_score_mean'] if x['weighted_score_mean'] is not None else 0),
                 reverse=True)
    
    # 保存 CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folder_name', 'weighted_score_mean'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"已提取 {len(results)} 条记录")
    print(f"结果已保存到: {output_csv}")
    print("\n提取结果:")
    print(f"{'folder_name':<25} {'weighted_score_mean':>20}")
    print("-" * 46)
    for row in results:
        print(f"{row['folder_name']:<25} {row['weighted_score_mean']:>20.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从 JSON 文件提取 weighted_score_mean 和 folder_name")
    parser.add_argument('--input', type=str, 
                       default='summary_all_results_alpha0.9.json',
                       help='输入的 JSON 文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出的 CSV 文件路径（可选）')
    
    args = parser.parse_args()
    
    # 确定 JSON 文件路径
    json_path = args.input
    if not os.path.isabs(json_path):
        # 如果是相对路径，相对于脚本所在目录
        script_dir = Path(__file__).parent.resolve()
        json_path = script_dir / json_path
    
    # 提取并保存
    extract_weighted_scores(str(json_path), args.output)
