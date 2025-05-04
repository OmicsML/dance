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
    """Helper function to convert a single value."""
    if isinstance(x, str):
        try:
            complex_val = complex(x.strip('()'))
            # If imaginary part is close to 0, return real part
            if abs(complex_val.imag) < 1e-10:
                return float(complex_val.real)
            return complex_val
        except ValueError:
            return x
    elif isinstance(x, complex):
        # If imaginary part is close to 0, return real part
        if abs(x.imag) < 1e-10:
            return float(x.real)
        return x
    return x


def unify_complex_float_types_cell(df):
    """Process by cell."""
    for col in df.columns:
        for idx in df.index:
            df.at[idx, col] = convert_complex_value(df.at[idx, col])
    return df


def unify_complex_float_types_row(df):
    """Process by row."""
    for idx in df.index:
        df.loc[idx] = df.loc[idx].apply(convert_complex_value)
    return df


def unify_complex_float_types(df):
    """Process by column."""
    for col in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Check if contains complex numbers
        has_complex = df[col].apply(lambda x: isinstance(x, complex)).any()

        if has_complex:
            # Convert column to complex and process
            df[col] = df[col].apply(convert_complex_value)

    return df


def process_excel_files(excel_files):
    # List to store all data
    all_data = []

    for file_path in excel_files:
        # Get filename (without extension)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Read all sheets in Excel file
        excel = pd.ExcelFile(file_path)

        # Process each sheet
        for sheet_name in excel.sheet_names:
            # Read data
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Transpose data
            df_transposed = df.transpose()

            # Add filename and sheet name columns
            df_transposed['file_name'] = file_name
            df_transposed['sheet_name'] = sheet_name

            # Add data to list
            all_data.append(df_transposed)

    # Merge all data
    final_df = pd.concat(all_data, ignore_index=True)

    # Unify data types
    final_df = unify_complex_float_types(final_df)

    # Save as CSV
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
            # df=unify_complex_float_types_row(df) #Some complex numbers may lose precision, but it's not a big issue since only real parts are used for comparison
            df = unify_complex_float_types_cell(
                df
            )  #Some complex numbers may lose precision, but it's not a big issue since only real parts are used for comparison
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
    print(f"Combined data has been saved to: {output_file}")
