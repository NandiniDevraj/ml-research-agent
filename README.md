# 🤖 Autonomous ML Research Agent

An AI agent system that acts as a junior ML engineer — 
give it a dataset and a problem, it autonomously runs 
the entire ML research lifecycle and deploys the best model.

## 🎯 What It Does

Upload a CSV dataset, define the problem, and watch 6 
specialized agents run the complete ML pipeline:

| Agent | Role |
|-------|------|
| 🔍 Data Explorer | Profiles dataset, detects issues |
| ⚙️ Feature Engineer | Engineers features, handles imbalance |
| 🧪 Experiment Agent | Trains 5 models, logs to MLflow |
| 🏆 Model Selector | Selects best model, registers in MLflow |
| 📝 Report Writer | Generates full PDF research report |
| 🚀 Deployment Agent | Deploys model as FastAPI endpoint |

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| API Docs | http://35.170.202.192:8000/docs |
| MLflow Dashboard | Run locally: `mlflow ui` |
| Streamlit UI | Run locally: `streamlit run app/streamlit_app.py` |

## 🛠️ Tech Stack

- **Agent Framework:** LangGraph + GPT-4o-mini
- **ML Models:** scikit-learn, XGBoost, LightGBM
- **Experiment Tracking:** MLflow
- **Cloud:** AWS S3 (artifact storage)
- **API:** FastAPI
- **UI:** Streamlit

## 🚀 Quick Start

1. Clone the repo
2. Create virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate
```
3. Install dependencies
```bash
   pip install -r requirements.txt
```
4. Create `.env` file
```
   OPENAI_API_KEY=your_key
   AWS_REGION=us-east-1
   S3_BUCKET=your_bucket
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   ARTIFACT_BUCKET=./mlflow-artifacts
```

5. Run the Streamlit UI
```bash
   streamlit run app/streamlit_app.py
```

## 📊 Results on CMS Hospital Readmission Dataset

| Model | AUC | F1 | Accuracy |
|-------|-----|----|----------|
| LightGBM | 0.9177 | 0.92 | 0.92 |
| XGBoost | 0.9152 | 0.91 | 0.91 |
| RandomForest | 0.9012 | 0.90 | 0.90 |
| NeuralNetwork | 0.8757 | 0.87 | 0.87 |
| LogisticRegression | 0.8006 | 0.80 | 0.80 |

## 🏗️ Architecture
```
CSV Upload → Streamlit UI
                ↓
        LangGraph Orchestrator
        ↙    ↓    ↓    ↓    ↓    ↘
    EDA  FE  Exp  Sel  Rep  Deploy
                ↓
        MLflow Tracking + Registry
                ↓
        AWS S3 Artifact Storage
                ↓
        FastAPI Prediction Endpoint
        (Live on AWS ECS Fargate)
```

## 👩‍💻 Author

Nandini Devaraj