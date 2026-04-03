# tools/feature_tools.py
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import json
import os

@tool
def handle_missing_values(file_path: str) -> str:
    """
    Handle missing values in the dataset.
    Numeric columns → filled with median.
    Categorical columns → filled with mode.
    Saves cleaned file as processed_step1.csv
    """
    try:
        df = pd.read_csv(file_path)
        report = {}

        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    report[col] = f"Filled {missing} missing values with median ({fill_value:.2f})"
                else:
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    report[col] = f"Filled {missing} missing values with mode ({fill_value})"

        out_path = os.path.join(os.path.dirname(file_path), "processed_step1.csv")
        df.to_csv(out_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": out_path,
            "actions_taken": report
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def encode_categorical_columns(file_path: str) -> str:
    """
    Encode categorical columns using Label Encoding.
    Saves result as processed_step2.csv.
    """
    try:
        df = pd.read_csv(file_path)
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if not categorical_cols:
            return "No categorical columns found — skipping encoding."

        encoding_map = {}
        le = LabelEncoder()

        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = "label encoded"

        out_path = file_path.replace("step1", "step2")
        df.to_csv(out_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": out_path,
            "encoded_columns": encoding_map
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def create_new_features(file_path: str) -> str:
    """
    Create new features from existing ones to help model performance.
    Saves result as processed_step3.csv.
    """
    try:
        df = pd.read_csv(file_path)
        new_features = []

        # Feature 1: medication intensity (medications per day)
        if "num_medications" in df.columns and "time_in_hospital" in df.columns:
            df["medication_intensity"] = (
                df["num_medications"] / df["time_in_hospital"].replace(0, 1)
            ).round(3)
            new_features.append("medication_intensity")

        # Feature 2: procedure burden (procedures + diagnoses combined)
        if "num_procedures" in df.columns and "num_diagnoses" in df.columns:
            df["procedure_burden"] = df["num_procedures"] + df["num_diagnoses"]
            new_features.append("procedure_burden")

        # Feature 3: is elderly flag
        if "age" in df.columns:
            df["is_elderly"] = (df["age"] >= 65).astype(int)
            new_features.append("is_elderly")

        # Feature 4: log transform for skewed medication column
        if "num_medications" in df.columns:
            df["log_num_medications"] = np.log1p(df["num_medications"]).round(3)
            new_features.append("log_num_medications")

        out_path = file_path.replace("step2", "step3")
        df.to_csv(out_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": out_path,
            "new_features_created": new_features,
            "total_features_now": len(df.columns)
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def apply_smote(file_path: str, target_column: str) -> str:
    """
    Apply SMOTE to fix class imbalance in the dataset.
    Saves result as processed_final.csv — this is the 
    file the Experiment Agent will use for training.
    """
    try:
        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            return f"Target column '{target_column}' not found."

        X = df.drop(columns=[target_column])
        y = df[target_column]

        before = y.value_counts().to_dict()

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled

        after = y_resampled.value_counts().to_dict()

        out_path = file_path.replace("step3", "final")
        df_resampled.to_csv(out_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": out_path,
            "class_distribution_before": {str(k): int(v) for k, v in before.items()},
            "class_distribution_after": {str(k): int(v) for k, v in after.items()},
            "rows_before": len(df),
            "rows_after": len(df_resampled)
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def validate_final_dataset(file_path: str) -> str:
    """
    Run a final validation check on the processed dataset.
    Confirms no missing values, all numeric, ready for training.
    """
    try:
        df = pd.read_csv(file_path)
        issues = []

        missing = df.isnull().sum().sum()
        if missing > 0:
            issues.append(f"Still has {missing} missing values")

        non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
        if non_numeric:
            issues.append(f"Non-numeric columns remaining: {non_numeric}")

        return json.dumps({
            "status": "ready" if not issues else "issues found",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "issues": issues if issues else "None — dataset is clean and ready for training"
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"