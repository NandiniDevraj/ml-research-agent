# agents/model_selector.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.selector_tools import (
    get_all_experiment_runs,
    register_best_model,
    get_registered_model_info
)

load_dotenv()


def run_model_selector(experiment_summary: str) -> dict:
    """
    Run the Model Selector Agent.
    Picks the best model and registers it in MLflow Model Registry.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    tools = [
        get_all_experiment_runs,
        register_best_model,
        get_registered_model_info
    ]

    agent = create_react_agent(llm, tools)

    prompt = f"""
    You are a Model Selector Agent — a senior ML engineer 
    responsible for selecting and registering the best model 
    for production deployment.

    The Experiment Agent ran 5 models and produced this summary:
    {experiment_summary}

    Your tasks:
    1. Call get_all_experiment_runs to fetch all MLflow runs
    2. Identify the best model based on AUC score (highest AUC wins)
    3. Call register_best_model with:
       - the run_id of the best model
       - model_name: "hospital-readmission-champion"
    4. Call get_registered_model_info to confirm registration

    Then write a Model Selection Report with:
    - Full comparison table of all 5 models
    - Why you selected the winning model (specific metrics)
    - Trade-offs considered (AUC vs F1 vs speed)
    - Confirmation the model is registered in MLflow Model Registry
    - The model URI for deployment
    """

    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    summary = result["messages"][-1].content
    return {"summary": summary}