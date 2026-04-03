# test_pipeline.py
import os
import boto3
from datetime import datetime
from dotenv import load_dotenv
from agents.data_explorer import run_data_explorer
from agents.feature_engineer import run_feature_engineer

load_dotenv()

def save_to_s3(content: str, s3_key: str) -> str:
    """Save any text content to S3."""
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    bucket = os.getenv("S3_BUCKET")
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=content.encode("utf-8"),
        ContentType="text/markdown"
    )
    return f"s3://{bucket}/{s3_key}"


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "hospital_readmission.csv")
    target_column = "readmitted"
    problem = "Predict 30-day hospital readmission risk"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Agent 1: Data Explorer ─────────────────────────────
    print("\n🔍 AGENT 1: Data Explorer running...")
    print("-" * 50)

    eda_report = run_data_explorer(
        file_path=file_path,
        target_column=target_column,
        problem_statement=problem
    )

    s3_path = save_to_s3(
        eda_report,
        f"reports/eda_report_{timestamp}.md"
    )
    print(eda_report)
    print(f"\n✅ EDA report saved → {s3_path}")

    # ── Agent 2: Feature Engineer ──────────────────────────
    print("\n\n⚙️  AGENT 2: Feature Engineer running...")
    print("-" * 50)

    fe_result = run_feature_engineer(
        file_path=file_path,
        target_column=target_column,
        eda_report=eda_report
    )

    s3_path = save_to_s3(
        fe_result["summary"],
        f"reports/feature_engineering_{timestamp}.md"
    )
    print(fe_result["summary"])
    print(f"\n✅ Feature engineering report saved → {s3_path}")
    print(f"✅ Final dataset ready at → {fe_result['final_dataset_path']}")

    from agents.experiment_agent import run_experiment_agent

    # ── Agent 3: Experiment Agent ──────────────────────────
    print("\n\n🧪 AGENT 3: Experiment Agent running...")
    print("-" * 50)

    exp_result = run_experiment_agent(
        file_path=fe_result["final_dataset_path"],
        target_column=target_column,
        feature_summary=fe_result["summary"]
    )

    s3_path = save_to_s3(
        exp_result["summary"],
        f"reports/experiment_summary_{timestamp}.md"
    )
    print(exp_result["summary"])
    print(f"\n✅ Experiment summary saved → {s3_path}")

    from agents.model_selector import run_model_selector

    # ── Agent 4: Model Selector ────────────────────────────
    print("\n\n🏆 AGENT 4: Model Selector running...")
    print("-" * 50)

    selector_result = run_model_selector(
        experiment_summary=exp_result["summary"]
    )

    s3_path = save_to_s3(
        selector_result["summary"],
        f"reports/model_selection_{timestamp}.md"
    )
    print(selector_result["summary"])
    print(f"\n✅ Model selection report saved → {s3_path}")

    from agents.report_writer import run_report_writer

    # ── Agent 5: Report Writer ─────────────────────────────
    print("\n\n📝 AGENT 5: Report Writer running...")
    print("-" * 50)

    report_result = run_report_writer(
        eda_report=eda_report,
        feature_summary=fe_result["summary"],
        experiment_summary=exp_result["summary"],
        selection_summary=selector_result["summary"],
        final_dataset_path=fe_result["final_dataset_path"]
    )

    # Upload PDF to S3
    import glob
    pdf_files = glob.glob("reports/*.pdf")
    if pdf_files:
        latest_pdf = max(pdf_files, key=os.path.getctime)
        bucket = os.getenv("S3_BUCKET")
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
        pdf_key = f"reports/{os.path.basename(latest_pdf)}"
        s3.upload_file(latest_pdf, bucket, pdf_key)
        print(f"\n✅ PDF report saved locally  → {latest_pdf}")
        print(f"✅ PDF report uploaded to S3 → s3://{bucket}/{pdf_key}")
    
    print(f"\n✅ Report summary saved → done")

    from agents.deployment_agent import run_deployment_agent

    # ── Agent 6: Deployment Agent ──────────────────────────
    print("\n\n🚀 AGENT 6: Deployment Agent running...")
    print("-" * 50)

    deploy_result = run_deployment_agent(
        final_dataset_path=fe_result["final_dataset_path"],
        selection_summary=selector_result["summary"]
    )

    s3_path = save_to_s3(
        deploy_result["summary"],
        f"reports/deployment_summary_{timestamp}.md"
    )
    print(deploy_result["summary"])
    print(f"\n✅ Deployment summary saved → {s3_path}")
    print("\n" + "=" * 60)
    print("🎉 ALL 6 AGENTS COMPLETE")
    print("=" * 60)
    print(f"\n→ Start your API with:")
    print(f"  cd api && uvicorn main:app --reload")
    print(f"\n→ Then open: http://localhost:8000/docs")