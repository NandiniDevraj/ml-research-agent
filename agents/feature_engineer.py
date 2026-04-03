# agents/feature_engineer.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.feature_tools import (
    handle_missing_values,
    encode_categorical_columns,
    create_new_features,
    apply_smote,
    validate_final_dataset
)

load_dotenv()


def run_feature_engineer(
    file_path: str,
    target_column: str,
    eda_report: str
) -> dict:
    """
    Run the Feature Engineer Agent.

    Args:
        file_path: path to the raw CSV
        target_column: column we want to predict
        eda_report: the report from the Data Explorer Agent

    Returns:
        dict with final dataset path and feature engineering summary
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    tools = [
        handle_missing_values,
        encode_categorical_columns,
        create_new_features,
        apply_smote,
        validate_final_dataset
    ]

    agent = create_react_agent(llm, tools)

    prompt = f"""
    You are a Feature Engineer Agent — a senior ML engineer preparing 
    data for model training.

    Dataset location: {file_path}
    Target variable: {target_column}

    The Data Explorer Agent already analyzed this dataset and found:
    {eda_report}

    Your job is to prepare the data for model training. Execute these 
    steps IN ORDER — each step's output file becomes the next step's input:

    Step 1: handle_missing_values → input: {file_path}
            outputs → processed_step1.csv

    Step 2: encode_categorical_columns → input: processed_step1.csv 
            outputs → processed_step2.csv

    Step 3: create_new_features → input: processed_step2.csv
            outputs → processed_step3.csv

    Step 4: apply_smote → input: processed_step3.csv, target: {target_column}
            outputs → processed_final.csv

    Step 5: validate_final_dataset → input: processed_final.csv
            Confirm dataset is clean and ready.

    After all steps, write a Feature Engineering Summary that includes:
    - What missing value strategy was used and why
    - Which columns were encoded
    - What new features were created and the reasoning behind each
    - Before/after class distribution from SMOTE
    - Final dataset dimensions
    - Confirmation that data is ready for model training
    """

    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    summary = result["messages"][-1].content

    # Build the final output path
    data_dir = os.path.dirname(file_path)
    final_path = os.path.join(data_dir, "processed_final.csv")

    return {
        "final_dataset_path": final_path,
        "summary": summary
    }