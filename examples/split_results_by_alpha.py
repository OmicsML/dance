"""将 combined_all_results_detail.csv 根据 alpha 值拆分成不同的表格 不包括 alpha=0.3."""
import csv
from pathlib import Path


def split_by_alpha(input_csv: str, output_dir: str = None):
    """根据 alpha 值拆分 CSV 文件.

    Args:
        input_csv: 输入的 combined_all_results_detail.csv 路径
        output_dir: 输出目录

    """
    # 读取数据
    data_by_alpha = {}  # {alpha: [rows]}

    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha = row['alpha']

            # 跳过 alpha=0.3
            if alpha == "0.3":
                continue

            if alpha not in data_by_alpha:
                data_by_alpha[alpha] = []

            data_by_alpha[alpha].append(row)

    if not data_by_alpha:
        print("没有找到需要拆分的数据（除了 alpha=0.3）")
        return

    # 获取输出目录
    if output_dir is None:
        output_dir = Path(input_csv).parent.resolve()
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 alpha 排序（降序）
    sorted_alphas = sorted(data_by_alpha.keys(), key=lambda x: float(x), reverse=True)

    print(f"将数据按 alpha 值拆分...")
    print(f"排除 alpha=0.3")
    print(f"生成 {len(sorted_alphas)} 个表格: {sorted_alphas}")

    for alpha in sorted_alphas:
        rows = data_by_alpha[alpha]
        output_file = output_dir / f"results_alpha{alpha}.csv"

        # 获取列名（排除 alpha 列）
        if rows:
            fieldnames = [k for k in rows[0].keys() if k != 'alpha']
            # 为每个 row 移除 alpha 键
            rows_clean = [{k: v for k, v in row.items() if k != 'alpha'} for row in rows]
        else:
            continue

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_clean)

        print(f"  ✓ {output_file.name} ({len(rows)} 行)")

    print(f"\n拆分完成！所有文件已保存到: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="根据 alpha 值拆分 CSV 文件")
    parser.add_argument('--input', type=str, default='combined_all_results_detail.csv', help='输入的 CSV 文件路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录（可选）')

    args = parser.parse_args()

    # 使用脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    input_csv = script_dir / args.input

    if not input_csv.exists():
        print(f"错误: 文件 {input_csv} 不存在")
        exit(1)

    split_by_alpha(str(input_csv), args.output_dir)
