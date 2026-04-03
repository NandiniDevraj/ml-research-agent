# tools/selector_tools.py
import mlflow
import mlflow.sklearn
import json
import os
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment("hospital-readmission")


@tool
def get_all_experiment_runs() -> str:
    """
    Fetch all MLflow runs from the hospital-readmission experiment.
    Returns each run's name, AUC, F1, accuracy and run_id.
    """
    try:
        setup_mlflow()
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name("hospital-readmission")
        if not experiment:
            return "No experiment found named hospital-readmission"

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.auc DESC"]
        )

        results = []
        for run in runs:
            results.append({
                "run_id":   run.info.run_id,
                "run_name": run.info.run_name,
                "auc":      run.data.metrics.get("auc", 0),
                "f1":       run.data.metrics.get("f1", 0),
                "accuracy": run.data.metrics.get("accuracy", 0),
                "precision":run.data.metrics.get("precision", 0),
                "recall":   run.data.metrics.get("recall", 0),
            })

        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def register_best_model(run_id: str, model_name: str) -> str:
    """
    Register the best model in the MLflow Model Registry.
    Promotes it to Staging status.

    Args:
        run_id: the MLflow run ID of the best model
        model_name: name to register the model under
    """
    try:
        setup_mlflow()
        client = mlflow.tracking.MlflowClient()

        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging"
        )

        return json.dumps({
            "status": "success",
            "model_name": model_name,
            "version": result.version,
            "stage": "Staging",
            "run_id": run_id,
            "model_uri": model_uri
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_registered_model_info(model_name: str) -> str:
    """
    Get info about a registered model from the MLflow Model Registry.
    """
    try:
        setup_mlflow()
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions(model_name)
        result = [{
            "name":    v.name,
            "version": v.version,
            "stage":   v.current_stage,
            "run_id":  v.run_id
        } for v in versions]

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"