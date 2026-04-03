# agents/experiment_agent.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.experiment_tools import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    train_neural_network
)

load_dotenv()


def run_experiment_agent(
    file_path: str,
    target_column: str,
    feature_summary: str
) -> dict:
    """
    Run the Experiment Agent — trains all models
    and returns a summary of all runs.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    tools = [
        train_logistic_regression,
        train_random_forest,
        train_xgboost,
        train_lightgbm,
        train_neural_network
    ]

    agent = create_react_agent(llm, tools)

    prompt = f"""
    You are an Experiment Agent — a senior ML engineer running 
    model training experiments.

    Dataset location: {file_path}
    Target variable: {target_column}

    The Feature Engineer prepared this dataset:
    {feature_summary}

    Your job is to train ALL 5 models on this dataset:
    1. train_logistic_regression
    2. train_random_forest
    3. train_xgboost
    4. train_lightgbm
    5. train_neural_network

    Train every single model — do not skip any.
    All results are automatically logged to MLflow.

    After all models are trained, write an Experiment Summary with:
    - A results table showing all 5 models and their AUC, F1, Accuracy
    - Which model performed best and why
    - Which model performed worst and why
    - Any interesting patterns you noticed across models
    - Your recommendation for which model to promote to production
    """

    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    summary = result["messages"][-1].content
    return {"summary": summary}