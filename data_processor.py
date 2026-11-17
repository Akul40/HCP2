
import pandas as pd
import re


def load_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Unable to read CSV: {e}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least two columns.")

    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    return df


def detect_type(series):
    values = series.dropna()

    if values.empty:
        return "Empty"

    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"

    dates = pd.to_datetime(values, errors="coerce")

    if dates.notna().mean() > 0.85:
        return "Date"

    return "Categorical"


def profile_data(df):

    columns = []
    possible_keys = []
    features = []

    key_pattern = re.compile(
        r"(hcp|npi|doctor|provider|physician|customer|account|^id$|_id$)",
        re.I
    )

    for col in df.columns:

        series = df[col]

        dtype = detect_type(series)

        is_key = bool(key_pattern.search(str(col)))

        missing = int(series.isna().sum())

        unique = int(series.nunique())

        if is_key:
            role = "Possible HCP Identifier"
            possible_keys.append(col)

        elif dtype in ["Numeric", "Categorical"]:
            role = "Potential Feature"

            if 1 < unique <= max(30, int(len(df) * 0.5)):
                features.append({
                    "column": col,
                    "type": dtype
                })

        else:
            role = "Review"

        columns.append({
            "column": col,
            "type": dtype,
            "missing": missing,
            "unique": unique,
            "role": role
        })

    missing_cells = int(df.isna().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    return {
        "rows": len(df),
        "total_columns": len(df.columns),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "possible_hcp_keys": possible_keys,
        "columns": columns,
        "features": features
    }
