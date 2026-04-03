# tools/experiment_tools.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import json
from langchain_core.tools import tool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score,
    accuracy_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from dotenv import load_dotenv

load_dotenv()


def get_data(file_path: str, target_column: str):
    """Helper — load and split dataset."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test) -> dict:
    """Helper — compute all metrics."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )
    return {
        "auc":       round(roc_auc_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4)
    }


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = ""
    experiment_name = "hospital-readmission"
    try:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=os.getenv("ARTIFACT_BUCKET")
        )
    except Exception:
        pass  # already exists
    mlflow.set_experiment(experiment_name)


@tool
def train_logistic_regression(file_path: str, target_column: str) -> str:
    """
    Train a Logistic Regression model and log results to MLflow.
    Returns metrics as JSON.
    """
    try:
        setup_mlflow()
        X_train, X_test, y_train, y_test = get_data(file_path, target_column)

        params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
        model = LogisticRegression(**params)

        with mlflow.start_run(run_name="LogisticRegression"):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

        return json.dumps({
            "model": "LogisticRegression",
            "run_id": run_id,
            "metrics": metrics
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def train_random_forest(file_path: str, target_column: str) -> str:
    """
    Train a Random Forest model and log results to MLflow.
    Returns metrics as JSON.
    """
    try:
        setup_mlflow()
        X_train, X_test, y_train, y_test = get_data(file_path, target_column)

        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestClassifier(**params)

        with mlflow.start_run(run_name="RandomForest"):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

        return json.dumps({
            "model": "RandomForest",
            "run_id": run_id,
            "metrics": metrics
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def train_xgboost(file_path: str, target_column: str) -> str:
    """
    Train an XGBoost model and log results to MLflow.
    Returns metrics as JSON.
    """
    try:
        setup_mlflow()
        X_train, X_test, y_train, y_test = get_data(file_path, target_column)

        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0
        }
        model = XGBClassifier(**params)

        with mlflow.start_run(run_name="XGBoost"):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

        return json.dumps({
            "model": "XGBoost",
            "run_id": run_id,
            "metrics": metrics
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def train_lightgbm(file_path: str, target_column: str) -> str:
    """
    Train a LightGBM model and log results to MLflow.
    Returns metrics as JSON.
    """
    try:
        setup_mlflow()
        X_train, X_test, y_train, y_test = get_data(file_path, target_column)

        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": -1
        }
        model = LGBMClassifier(**params)

        with mlflow.start_run(run_name="LightGBM"):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

        return json.dumps({
            "model": "LightGBM",
            "run_id": run_id,
            "metrics": metrics
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def train_neural_network(file_path: str, target_column: str) -> str:
    """
    Train a Neural Network (MLP) model and log results to MLflow.
    Returns metrics as JSON.
    """
    try:
        setup_mlflow()
        X_train, X_test, y_train, y_test = get_data(file_path, target_column)

        params = {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 300,
            "random_state": 42
        }
        model = MLPClassifier(**params)

        with mlflow.start_run(run_name="NeuralNetwork"):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params({"hidden_layers": "64,32", "max_iter": 300})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

        return json.dumps({
            "model": "NeuralNetwork",
            "run_id": run_id,
            "metrics": metrics
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"