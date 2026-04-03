# tools/data_tools.py
# These are the functions our Data Explorer Agent can call.
# Each @tool decorator turns a normal Python function into 
# something the LLM can decide to use.

import pandas as pd
import numpy as np
from langchain_core.tools import tool
import json

@tool
def profile_dataset(file_path: str) -> str:
    """
    Load a CSV dataset and return its basic profile:
    shape, column names, data types, memory usage.
    """
    try:
        df = pd.read_csv(file_path)
        profile = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return json.dumps(profile, indent=2)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


@tool
def check_missing_values(file_path: str) -> str:
    """
    Analyze missing values in the dataset.
    Returns count and percentage of missing values per column.
    """
    try:
        df = pd.read_csv(file_path)
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        result = {
            col: {
                "count": int(missing[col]),
                "percentage": float(missing_pct[col])
            }
            for col in df.columns if missing[col] > 0
        }
        if not result:
            return "No missing values found in the dataset."
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def analyze_numeric_features(file_path: str) -> str:
    """
    Analyze numeric features: mean, std, min, max, skewness.
    """
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return "No numeric columns found."
        stats = df[numeric_cols].describe().round(3).to_dict()
        skewness = df[numeric_cols].skew().round(3).to_dict()
        result = {
            "numeric_columns": numeric_cols,
            "statistics": stats,
            "skewness": skewness
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def check_class_balance(file_path: str, target_column: str) -> str:
    """
    Check the class distribution of the target variable.
    Flags if the dataset is imbalanced (minority class < 20%).
    """
    try:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            available = df.columns.tolist()
            return f"Column '{target_column}' not found. Available: {available}"
        counts = df[target_column].value_counts()
        percentages = (counts / len(df) * 100).round(2)
        result = {
            "class_counts": counts.to_dict(),
            "class_percentages": percentages.to_dict(),
            "is_imbalanced": bool(percentages.min() < 20)
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def detect_outliers(file_path: str) -> str:
    """
    Detect outliers in numeric columns using the IQR method.
    Returns columns with outliers and how many rows are affected.
    """
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_report = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[
                (df[col] < Q1 - 1.5 * IQR) |
                (df[col] > Q3 + 1.5 * IQR)
            ]
            if len(outliers) > 0:
                outlier_report[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": round(len(outliers) / len(df) * 100, 2)
                }
        if not outlier_report:
            return "No significant outliers detected."
        return json.dumps(outlier_report, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"