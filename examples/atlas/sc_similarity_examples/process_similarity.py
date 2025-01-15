import glob
import os

import numpy as np
import pandas as pd


def unify_complex_float_types(df):
    for col in df.columns:
        # 跳过非数值列
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # 检查是否包含复数
        has_complex = df[col].apply(lambda x: isinstance(x, complex)).any()

        if has_complex:
            # 将列转换为复数类型
            complex_series = df[col].apply(lambda x: complex(x) if not isinstance(x, complex) else x)

            # 检查是否所有虚部都为0
            all_imag_zero = all(abs(x.imag) < 1e-10 for x in complex_series)

            if all_imag_zero:
                # 如果所有虚部都为0，转换为浮点数
                df[col] = complex_series.apply(lambda x: float(x.real))
            else:
                # 保持为复数
                df[col] = complex_series

    return df


def process_excel_files(folder_path):
    # 存储所有数据的列表
    all_data = []

    # 获取文件夹中所有的xlsx文件
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

    for file_path in excel_files:
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # 读取Excel文件中的所有表
        excel = pd.ExcelFile(file_path)

        # 处理每个表
        for sheet_name in excel.sheet_names:
            # 读取数据
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 转置数据
            df_transposed = df.transpose()

            # 添加文件名和表名列
            df_transposed['文件名'] = file_name
            df_transposed['表名'] = sheet_name

            # 将数据添加到列表中
            all_data.append(df_transposed)

    # 合并所有数据
    final_df = pd.concat(all_data, ignore_index=True)

    # 统一数据类型
    final_df = unify_complex_float_types(final_df)

    # 保存为CSV
    output_path = os.path.join(folder_path, 'combined_output.csv')
    final_df.to_csv(output_path, encoding='utf-8-sig', index=True)

    return output_path


# 使用示例
folder_path = "你的文件夹路径"  # 替换为实际的文件夹路径
output_file = process_excel_files(folder_path)
print(f"已将合并后的数据保存到: {output_file}")
