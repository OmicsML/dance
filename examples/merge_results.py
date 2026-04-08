"""将 graph_automl 的 summary_all_weighted_scores.csv 和 evolo 的 all_combined_scores.csv
综合起来."""
import csv
from pathlib import Path


def normalize_folder_name(name: str) -> str:
    """标准化 folder_name（统一大小写）"""
    # SpaGRA 特殊情况：统一为小写 spagra
    if name.lower() == 'domain_spagra' or name.lower() == 'domain_spagra':
        return 'domain_spagra'
    return name.lower()


def merge_csv_files(automl_csv: str, evolo_csv: str, output_csv: str):
    """综合两个 CSV 文件.

    Args:
        automl_csv: graph_automl/summary_all_weighted_scores.csv 路径
        evolo_csv: evolo/all_combined_scores.csv 路径
        output_csv: 输出的综合 CSV 路径

    """
    # 读取并处理 evolo 数据
    evolo_reader_data = {}
    # 使用字典去重：(alpha, folder_name) 保留 test_combined_score 最大的

    with open(evolo_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 标准化 folder_name 作为 key
            folder_name = normalize_folder_name(row['folder_name'].strip())
            alpha = float(row['alpha'].strip())
            is_evoloved = row['is_evoloved'].strip()
            score = float(row['test_combined_score'])

            # 映射 is_evoloved 到 method
            method = "Full" if is_evoloved == "True" else "baseline"

            key = (alpha, folder_name, method)

            # 去重：保留分数最大的
            if key not in evolo_reader_data or score > evolo_reader_data[key]['test_combined_score']:
                evolo_reader_data[key] = {
                    'alpha': alpha,
                    'folder_name': folder_name,
                    'test_combined_score': score,
                    'method': method
                }

    evolo_reader_data = list(evolo_reader_data.values())

    # 读取并处理 automl 数据（需要与 evolo 数据合并，需要去重）
    automl_data_dict = {}

    with open(automl_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder_name = normalize_folder_name(row['folder_name'].strip())
            alpha = float(row['alpha'].strip()) if row['alpha'].strip() else None
            weighted_score_mean = row['weighted_score_mean']
            score = float(weighted_score_mean) if weighted_score_mean else None

            if score is None:
                continue

            key = (alpha, folder_name, 'automl')

            # 去重：保留分数最大的
            if key not in automl_data_dict or score > automl_data_dict[key]['test_combined_score']:
                automl_data_dict[key] = {
                    'alpha': alpha,
                    'folder_name': folder_name,
                    'test_combined_score': score,
                    'method': 'automl'
                }

    automl_data = list(automl_data_dict.values())

    # 合并数据
    all_data = evolo_reader_data + automl_data

    # 按 method、alpha 降序、test_combined_score 降序排序
    # method 优先级: automl > Full > baseline
    method_priority = {'automl': 0, 'Full': 1, 'baseline': 2}
    all_data.sort(key=lambda x: (method_priority.get(x['method'], 3), -x['alpha']
                                 if x['alpha'] else 0, -x['test_combined_score'] if x['test_combined_score'] else 0))

    # 写入输出文件
    fieldnames = ['alpha', 'folder_name', 'test_combined_score', 'method']

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    # 统计信息
    evolo_count = len(evolo_reader_data)
    automl_count = len(automl_data)
    total_count = len(all_data)

    # 按 method 分组统计
    method_counts = {}
    for row in all_data:
        method = row['method']
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"综合完成！")
    print(f"  - evolo 数据: {evolo_count} 条")
    print(f"  - automl 数据: {automl_count} 条")
    print(f"  - 总计: {total_count} 条")
    print(f"\n各 method 数量:")
    for method, count in sorted(method_counts.items()):
        print(f"  - {method}: {count}")
    print(f"\n结果已保存到: {output_csv}")

    # 显示部分结果预览
    print("\n结果预览（前 15 行）:")
    print(f"{'alpha':<8} {'folder_name':<25} {'test_combined_score':>22} {'method':<12}")
    print("-" * 70)
    for i, row in enumerate(all_data[:15]):
        alpha_str = f"{row['alpha']:.1f}" if row['alpha'] else "N/A"
        score_str = f"{row['test_combined_score']:.4f}" if row['test_combined_score'] else "N/A"
        print(f"{alpha_str:<8} {row['folder_name']:<25} {score_str:>22} {row['method']:<12}")

    return all_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="综合 graph_automl 和 evolo 的 CSV 文件")
    parser.add_argument('--automl', type=str, default='summary_all_weighted_scores.csv', help='graph_automl 的 CSV 文件路径')
    parser.add_argument('--evolo', type=str, default='all_combined_scores.csv', help='evolo 的 CSV 文件路径')
    parser.add_argument('--output', type=str, default='combined_all_results.csv', help='输出的综合 CSV 文件路径')

    args = parser.parse_args()

    # 使用脚本所在目录
    script_dir = Path(__file__).parent.resolve()

    automl_csv = script_dir / args.automl
    evolo_csv = script_dir / args.evolo
    output_csv = script_dir / args.output

    merge_csv_files(str(automl_csv), str(evolo_csv), str(output_csv))
