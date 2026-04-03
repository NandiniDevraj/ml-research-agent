# agents/deployment_agent.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.deployment_tools import (
    load_champion_model,
    get_model_feature_info,
    create_fastapi_server,
    verify_api_files
)

load_dotenv()


def run_deployment_agent(
    final_dataset_path: str,
    selection_summary: str
) -> dict:
    """
    Run the Deployment Agent.
    Loads champion model and generates FastAPI server.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    tools = [
        load_champion_model,
        get_model_feature_info,
        create_fastapi_server,
        verify_api_files
    ]

    agent = create_react_agent(llm, tools)

    prompt = f"""
    You are a Deployment Agent — an ML engineer deploying 
    a trained model to production as a REST API.

    The Model Selector registered this champion model:
    {selection_summary}

    Final dataset location: {final_dataset_path}

    Execute these steps IN ORDER:

    Step 1: Call load_champion_model
            → loads model from MLflow, saves to api/champion_model.pkl

    Step 2: Call get_model_feature_info with {final_dataset_path}
            → gets feature names and a sample input

    Step 3: Call create_fastapi_server with the features_json 
            from step 2 (pass the FULL JSON string returned)
            → generates the FastAPI server code

    Step 4: Call verify_api_files
            → confirms everything is in place

    Then write a Deployment Summary with:
    - What model was deployed and its performance metrics
    - All available API endpoints with descriptions
    - A sample curl command to call /predict
    - A sample Python request to call /predict
    - Instructions to start the server
    """

    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    return {"summary": result["messages"][-1].content}