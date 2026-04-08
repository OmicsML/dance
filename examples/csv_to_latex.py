#!/usr/bin/env python3
"""CSV to LaTeX Table Converter for Overleaf 快速将CSV文件转换为Overleaf可用的LaTeX表格格式.

使用方法:     python csv_to_latex.py your_file.csv     python csv_to_latex.py your_file.csv
--output your_output.tex     python csv_to_latex.py your_file.csv --decimal-align  #
数值按小数点对齐

"""

import argparse
import csv
import os
import sys
from pathlib import Path


def read_csv(filepath):
    """读取CSV文件."""
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def count_decimal_places(value):
    """计算数值的小数位数."""
    try:
        if '.' in str(value):
            return len(str(value).split('.')[1])
        return 0
    except:
        return 0


def format_cell(value, col_idx, decimal_align=False, col_widths=None):
    """格式化单个单元格."""
    value = str(value).strip()
    if not value:
        return ''

    if decimal_align and col_widths and col_idx > 0:
        # 数值列：按小数点对齐
        try:
            num = float(value)
            # 保留最多4位小数
            formatted = f"{num:.4f}".rstrip('0').rstrip('.')
            return formatted
        except ValueError:
            pass

    return value


def calculate_column_widths(rows):
    """计算每列的最大宽度."""
    if not rows:
        return []

    num_cols = len(rows[0])
    col_widths = [0] * num_cols

    for row in rows:
        for i, cell in enumerate(row):
            cell = str(cell).strip()
            width = len(cell)
            # 对于数值，估算显示宽度
            try:
                if '.' in cell:
                    width = max(len(cell.split('.')[0]), 8)
            except:
                pass
            col_widths[i] = max(col_widths[i], width)

    return col_widths


def csv_to_latex(rows, decimal_align=False, caption='', label='', bold_max_cols=None, escape_chars=True):
    """将CSV数据转换为LaTeX表格代码.

    Args:
        rows: CSV数据（列表的列表）
        decimal_align: 是否按小数点对齐数值
        caption: 表格标题
        label: 表格标签
        bold_max_cols: 需要高亮最大值的列索引列表（第一列除外）
        escape_chars: 是否转义LaTeX特殊字符

    """
    if not rows:
        return ''

    # 计算列宽
    col_widths = calculate_column_widths(rows)
    num_cols = len(rows[0])

    # LaTeX特殊字符转义
    def escape(text):
        if not escape_chars:
            return text
        text = str(text)
        replacements = [
            ('&', r'\&'),
            ('%', r'\%'),
            ('$', r'\$'),
            ('#', r'\#'),
            ('_', r'\_'),
            ('{', r'\{'),
            ('}', r'\}'),
            ('~', r'\textasciitilde'),
            ('^', r'\textasciicircum'),
            ('\\', r'\textbackslash'),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    # 找到每行的最大值位置（排除第一列的Method名称）
    # max_values_per_row[row_idx] = col_idx_of_max_value
    max_values_per_row = {}
    if bold_max_cols:
        for row_idx in range(1, len(rows)):
            row = rows[row_idx]
            max_val = float('-inf')
            max_col_idx = -1
            # 从第二列开始找（跳过Method列）
            for col_idx in range(1, len(row)):
                try:
                    val = float(row[col_idx])
                    if val > max_val:
                        max_val = val
                        max_col_idx = col_idx
                except:
                    pass
            if max_col_idx >= 0:
                max_values_per_row[row_idx] = max_col_idx

    # 构建表格
    lines = []

    # 表格环境
    if caption or label:
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')

    # 计算列格式说明符
    col_format = 'l' + 'c' * (num_cols - 1) if decimal_align else 'l' * num_cols

    # 使用booktabs的三线表
    lines.append(r'\begin{tabular}{' + col_format + '}')
    lines.append(r'\toprule')

    # 表头
    header = rows[0]
    header_cells = []
    for i, cell in enumerate(header):
        cell = escape(cell)
        # 替换希腊字母等
        cell = cell.replace('α', r'$\alpha$')
        header_cells.append(cell)
    lines.append(' & '.join(header_cells) + r' \\')
    lines.append(r'\midrule')

    # 数据行
    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        cells = []
        for col_idx, cell in enumerate(row):
            cell = escape(str(cell).strip())

            # 高亮该行的最大值
            if row_idx in max_values_per_row and col_idx == max_values_per_row[row_idx] and cell:
                try:
                    float(cell)  # 验证是数值
                    cell = r'\textbf{' + cell + '}'
                except:
                    pass

            cells.append(cell)
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    if caption:
        lines.append(r'\caption{' + escape(caption) + '}')
    if label:
        lines.append(r'\label{tab:' + escape(label) + '}')

    if caption or label:
        lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='快速将CSV转换为Overleaf LaTeX表格', formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
示例:
    python csv_to_latex.py data.csv
    python csv_to_latex.py data.csv -o table.tex
    python csv_to_latex.py data.csv --decimal-align
    python csv_to_latex.py data.csv --caption "实验结果" --label "exp1"
        """)

    parser.add_argument('csv_file', nargs='?', help='输入的CSV文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径 (默认: stdout)')
    parser.add_argument('--decimal-align', action='store_true', help='数值按小数点对齐')
    parser.add_argument('--caption', default='', help='表格标题')
    parser.add_argument('--label', default='', help='表格标签')
    parser.add_argument('--bold-max', action='store_true', help='高亮每列最大值')
    parser.add_argument('--booktabs', action='store_true', default=True, help='使用booktabs三线表样式 (默认开启)')

    args = parser.parse_args()

    if not args.csv_file:
        parser.print_help()
        print("\n" + "=" * 50)
        print("快速使用：拖拽CSV文件到终端运行")
        print("=" * 50)
        return

    if not os.path.exists(args.csv_file):
        print(f"错误: 文件不存在: {args.csv_file}")
        sys.exit(1)

    # 读取CSV
    rows = read_csv(args.csv_file)

    if not rows:
        print("错误: CSV文件为空")
        sys.exit(1)

    # 转换为LaTeX
    bold_cols = list(range(1, len(rows[0]))) if args.bold_max else None

    latex_code = csv_to_latex(rows, decimal_align=args.decimal_align, caption=args.caption, label=args.label,
                              bold_max_cols=bold_cols)

    # 输出
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        print(f"✓ 已保存到: {args.output}")
    else:
        print("\n" + "=" * 50)
        print("生成的LaTeX代码:")
        print("=" * 50)
        print(latex_code)
        print("=" * 50)


if __name__ == '__main__':
    main()
