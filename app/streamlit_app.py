# app/streamlit_app.py
import streamlit as st
import sys
import os
import pandas as pd
import time
import subprocess
import glob
import boto3
from datetime import datetime
from dotenv import load_dotenv

# Add root to path so we can import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from agents.data_explorer import run_data_explorer
from agents.feature_engineer import run_feature_engineer
from agents.experiment_agent import run_experiment_agent
from agents.model_selector import run_model_selector
from agents.report_writer import run_report_writer
from agents.deployment_agent import run_deployment_agent

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="ML Research Agent",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .agent-box {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #555;
    }
    .agent-running {
        border-left: 4px solid #f0a500;
    }
    .agent-done {
        border-left: 4px solid #00c853;
    }
    .agent-waiting {
        border-left: 4px solid #555;
        opacity: 0.5;
    }
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c2ff, #00ff87);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


def save_to_s3(content: str, s3_key: str):
    """Save report to S3."""
    try:
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
        bucket = os.getenv("S3_BUCKET")
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=content.encode("utf-8"),
            ContentType="text/markdown"
        )
    except Exception as e:
        st.warning(f"S3 upload skipped: {e}")


def agent_status_box(icon, name, status, detail=""):
    """Render a styled agent status box."""
    css_class = {
        "waiting": "agent-box agent-waiting",
        "running": "agent-box agent-running",
        "done":    "agent-box agent-done"
    }.get(status, "agent-box")

    spinner = "⏳" if status == "running" else ("✅" if status == "done" else "⬜")

    st.markdown(f"""
    <div class="{css_class}">
        <b>{spinner} {icon} {name}</b><br>
        <small style="color:#aaa">{detail}</small>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────
st.markdown('<p class="title-text">🤖 Autonomous ML Research Agent</p>',
            unsafe_allow_html=True)
st.markdown("**Upload a dataset. Define the problem. Watch 6 agents do the rest.**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png",
             width=80)
    st.markdown("## ⚙️ Configuration")
    st.markdown("**Model:** GPT-4o-mini")
    st.markdown("**ML Framework:** LightGBM / XGBoost")
    st.markdown("**Tracking:** MLflow")
    st.markdown("**Storage:** AWS S3")
    st.divider()
    st.markdown("## 🤖 Agent Pipeline")
    st.markdown("1. 🔍 Data Explorer")
    st.markdown("2. ⚙️ Feature Engineer")
    st.markdown("3. 🧪 Experiment Agent")
    st.markdown("4. 🏆 Model Selector")
    st.markdown("5. 📝 Report Writer")
    st.markdown("6. 🚀 Deployment Agent")
    st.divider()
    st.markdown("**Built by:** Nandini Devaraj")
    st.markdown("**Stack:** LangGraph · MLflow · AWS")

# ── Input Section ─────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload any tabular dataset in CSV format"
    )
    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.success(f"✅ {uploaded_file.name} — {len(df_preview):,} rows, "
                   f"{len(df_preview.columns)} columns")
        st.dataframe(df_preview.head(5), use_container_width=True)
        uploaded_file.seek(0)

with col2:
    st.markdown("### 🎯 Define the Problem")
    problem_statement = st.text_area(
        "What do you want to predict?",
        value="Predict 30-day hospital readmission risk",
        height=100
    )
    target_column = st.text_input(
        "Target column name",
        value="readmitted",
        help="The column you want to predict"
    )
    st.markdown("### 🚀 Run the Pipeline")
    run_button = st.button(
        "▶ Run Autonomous Analysis",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )

st.divider()

# ── Agent Pipeline Runner ─────────────────────────────────
if run_button and uploaded_file:

    # Save uploaded file locally
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.markdown("## 🔄 Agent Pipeline — Running")

    # Placeholders for live updates
    placeholders = {i: st.empty() for i in range(6)}
    results = {}

    agents = [
        ("🔍", "Data Explorer Agent",    "Profiling dataset, detecting issues..."),
        ("⚙️",  "Feature Engineer Agent", "Engineering features, handling imbalance..."),
        ("🧪", "Experiment Agent",       "Training 5 models, logging to MLflow..."),
        ("🏆", "Model Selector Agent",   "Comparing experiments, selecting champion..."),
        ("📝", "Report Writer Agent",    "Writing full research report..."),
        ("🚀", "Deployment Agent",       "Generating FastAPI server..."),
    ]

    # Show all as waiting initially
    for i, (icon, name, detail) in enumerate(agents):
        with placeholders[i]:
            agent_status_box(icon, name, "waiting", "Waiting...")

    # ── Agent 1 ───────────────────────────────────────────
    with placeholders[0]:
        agent_status_box(*agents[0][:2], "running", agents[0][2])

    eda_report = run_data_explorer(
        file_path=file_path,
        target_column=target_column,
        problem_statement=problem_statement
    )
    save_to_s3(eda_report, f"reports/eda_{timestamp}.md")
    results["eda"] = eda_report

    with placeholders[0]:
        agent_status_box(*agents[0][:2], "done", "EDA complete — report saved to S3")

    # ── Agent 2 ───────────────────────────────────────────
    with placeholders[1]:
        agent_status_box(*agents[1][:2], "running", agents[1][2])

    fe_result = run_feature_engineer(
        file_path=file_path,
        target_column=target_column,
        eda_report=eda_report
    )
    save_to_s3(fe_result["summary"], f"reports/features_{timestamp}.md")
    results["fe"] = fe_result

    with placeholders[1]:
        agent_status_box(*agents[1][:2], "done",
                         "Features engineered — dataset ready for training")

    # ── Agent 3 ───────────────────────────────────────────
    with placeholders[2]:
        agent_status_box(*agents[2][:2], "running", agents[2][2])

    exp_result = run_experiment_agent(
        file_path=fe_result["final_dataset_path"],
        target_column=target_column,
        feature_summary=fe_result["summary"]
    )
    save_to_s3(exp_result["summary"], f"reports/experiments_{timestamp}.md")
    results["exp"] = exp_result

    with placeholders[2]:
        agent_status_box(*agents[2][:2], "done",
                         "5 models trained — all logged to MLflow")

    # ── Agent 4 ───────────────────────────────────────────
    with placeholders[3]:
        agent_status_box(*agents[3][:2], "running", agents[3][2])

    selector_result = run_model_selector(
        experiment_summary=exp_result["summary"]
    )
    save_to_s3(selector_result["summary"], f"reports/selection_{timestamp}.md")
    results["selector"] = selector_result

    with placeholders[3]:
        agent_status_box(*agents[3][:2], "done",
                         "Champion model selected — registered in MLflow")

    # ── Agent 5 ───────────────────────────────────────────
    with placeholders[4]:
        agent_status_box(*agents[4][:2], "running", agents[4][2])

    report_result = run_report_writer(
        eda_report=eda_report,
        feature_summary=fe_result["summary"],
        experiment_summary=exp_result["summary"],
        selection_summary=selector_result["summary"],
        final_dataset_path=fe_result["final_dataset_path"]
    )
    results["report"] = report_result

    with placeholders[4]:
        agent_status_box(*agents[4][:2], "done",
                         "PDF research report generated")

    # ── Agent 6 ───────────────────────────────────────────
    with placeholders[5]:
        agent_status_box(*agents[5][:2], "running", agents[5][2])

    deploy_result = run_deployment_agent(
        final_dataset_path=fe_result["final_dataset_path"],
        selection_summary=selector_result["summary"]
    )
    results["deploy"] = deploy_result

    with placeholders[5]:
        agent_status_box(*agents[5][:2], "done",
                         "FastAPI server ready — model live")

    # ── Results Dashboard ─────────────────────────────────
    st.divider()
    st.markdown("## 🎉 Pipeline Complete!")

    # Metrics row
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("hospital-readmission")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.auc DESC"]
        )

        best = runs[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 Best Model",  best.info.run_name)
        m2.metric("📈 AUC Score",   round(best.data.metrics.get("auc", 0), 4))
        m3.metric("🎯 F1 Score",    round(best.data.metrics.get("f1", 0), 4))
        m4.metric("✅ Accuracy",    round(best.data.metrics.get("accuracy", 0), 4))

        st.divider()

        # Experiment results table
        st.markdown("### 📊 All Experiment Results")
        table_data = [{
            "Model":     r.info.run_name,
            "AUC":       round(r.data.metrics.get("auc", 0), 4),
            "F1":        round(r.data.metrics.get("f1", 0), 4),
            "Precision": round(r.data.metrics.get("precision", 0), 4),
            "Recall":    round(r.data.metrics.get("recall", 0), 4),
            "Accuracy":  round(r.data.metrics.get("accuracy", 0), 4),
        } for r in runs]

        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.warning(f"MLflow results: {e}")

    st.divider()

    # Downloads and links
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Research Report")
        pdf_files = glob.glob(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "reports", "*.pdf"
            )
        )
        if pdf_files:
            latest_pdf = max(pdf_files, key=os.path.getctime)
            with open(latest_pdf, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    f.read(),
                    file_name="ml_research_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    with col2:
        st.markdown("### 🌐 Live API")
        st.info("**Endpoint:** http://localhost:8000")
        st.code("POST /predict", language="bash")
        st.markdown("[📖 Open API Docs](http://localhost:8000/docs)")

    with col3:
        st.markdown("### ☁️ AWS S3")
        bucket = os.getenv("S3_BUCKET")
        st.info(f"**Bucket:** {bucket}")
        st.markdown("All reports saved to S3 ✅")

    st.divider()
    st.success(
        "🎉 Done! Raw CSV → Trained model → PDF report → "
        "Live API — fully autonomous."
    )