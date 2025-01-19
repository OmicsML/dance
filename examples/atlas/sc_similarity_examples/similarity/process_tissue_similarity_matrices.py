import ast
import os

import numpy as np
import pandas as pd

from dance.settings import SIMILARITYDIR


def convert_to_complex(s):
    """Convert string representations of complex numbers to float values.

    Parameters
    ----------
    s : str or float
        Input value to convert

    Returns
    -------
    float
        Real part of complex number or NaN if conversion fails

    """
    if isinstance(s, float) or isinstance(s, int):
        return s
    try:
        s = ast.literal_eval(s)
        return float(s.real)
    except (ValueError, SyntaxError):
        return np.nan


def convert_complex_value(x):
    """转换单个值的辅助函数."""
    if isinstance(x, str):
        try:
            complex_val = complex(x.strip('()'))
            # 如果虚部接近0，返回实部
            if abs(complex_val.imag) < 1e-10:
                return float(complex_val.real)
            return complex_val
        except ValueError:
            return x
    elif isinstance(x, complex):
        # 如果虚部接近0，返回实部
        if abs(x.imag) < 1e-10:
            return float(x.real)
        return x
    return x


def unify_complex_float_types_cell(df):
    """按单元格处理."""
    for col in df.columns:
        for idx in df.index:
            df.at[idx, col] = convert_complex_value(df.at[idx, col])
    return df


def unify_complex_float_types_row(df):
    """按行处理."""
    for idx in df.index:
        df.loc[idx] = df.loc[idx].apply(convert_complex_value)
    return df


def unify_complex_float_types(df):
    """按列处理."""
    for col in df.columns:
        # 跳过非数值列
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # 检查是否包含复数
        has_complex = df[col].apply(lambda x: isinstance(x, complex)).any()

        if has_complex:
            # 将列转换为复数并处理
            df[col] = df[col].apply(convert_complex_value)

    return df


def process_excel_files(excel_files):
    # 存储所有数据的列表
    all_data = []

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
            df_transposed['file_name'] = file_name
            df_transposed['sheet_name'] = sheet_name

            # 将数据添加到列表中
            all_data.append(df_transposed)

    # 合并所有数据
    final_df = pd.concat(all_data, ignore_index=True)

    # 统一数据类型
    final_df = unify_complex_float_types(final_df)

    # 保存为CSV
    output_path = os.path.join(os.path.dirname(excel_files[0]), 'combined_output.csv')
    final_df.to_csv(output_path, encoding='utf-8-sig', index=True)

    return output_path


if __name__ == "__main__":
    tissues = ["blood", "brain", "heart", "intestine", "kidney", "lung", "pancreas"]
    for tissue in tissues:
        file_path = SIMILARITYDIR / f"data/dataset_similarity/{tissue}_similarity.xlsx"
        excel = pd.ExcelFile(file_path)
        for sheet_name in excel.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
            df = df[~df.index.duplicated(keep='last')]
            # df=unify_complex_float_types_row(df) #TODO 导致一些复数失真，但因为比较的时候只用实部，问题不大
            df = unify_complex_float_types_cell(df)  #TODO 导致一些复数失真，但因为比较的时候只用实部，问题不大
            if os.path.exists(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx"):
                mode = 'a'
                if_sheet_exists = "replace"
            else:
                mode = 'w'
                if_sheet_exists = None
            with pd.ExcelWriter(SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx", engine='openpyxl', mode=mode,
                                if_sheet_exists=if_sheet_exists) as writer:
                df.to_excel(writer, sheet_name=sheet_name)
    excel_files = [SIMILARITYDIR / f"data/new_sim/{tissue}_similarity.xlsx" for tissue in tissues]
    output_file = process_excel_files(excel_files)
    print(f"已将合并后的数据保存到: {output_file}")
