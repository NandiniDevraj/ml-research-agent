# agents/report_writer.py
import os
import boto3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.report_tools import (
    get_experiment_results,
    get_dataset_stats,
    save_report_as_markdown,
    convert_markdown_to_pdf
)

load_dotenv()


def run_report_writer(
    eda_report: str,
    feature_summary: str,
    experiment_summary: str,
    selection_summary: str,
    final_dataset_path: str
) -> dict:
    """
    Run the Report Writer Agent.
    Generates a complete PDF research report.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    tools = [
        get_experiment_results,
        get_dataset_stats,
        save_report_as_markdown,
        convert_markdown_to_pdf
    ]

    agent = create_react_agent(llm, tools)

    prompt = f"""
    You are a Report Writer Agent — a senior ML researcher 
    writing a complete research report for stakeholders.

    You have access to summaries from all previous agents:

    EDA REPORT:
    {eda_report}

    FEATURE ENGINEERING SUMMARY:
    {feature_summary}

    EXPERIMENT SUMMARY:
    {experiment_summary}

    MODEL SELECTION SUMMARY:
    {selection_summary}

    FINAL DATASET: {final_dataset_path}

    Your tasks:
    1. Call get_experiment_results to get exact metrics from MLflow
    2. Call get_dataset_stats to get final dataset dimensions
    3. Write a COMPLETE research report in markdown format
    4. Call save_report_as_markdown with the full report
    5. Call convert_markdown_to_pdf with the saved markdown path

    The report MUST include these sections in this order:
    
    # ML Research Report — Hospital Readmission Prediction
    
    ## Executive Summary
    (3-4 sentences: problem, approach, best result, recommendation)
    
    ## 1. Problem Statement
    (what we're predicting and why it matters)
    
    ## 2. Dataset Overview
    (rows, columns, source, key characteristics)
    
    ## 3. Data Quality Findings
    (missing values, outliers, class imbalance — what was found)
    
    ## 4. Feature Engineering
    (what was done to prepare data, what new features were created)
    
    ## 5. Experiment Results
    (table with all 5 models and their AUC, F1, Precision, Recall)
    
    ## 6. Model Selection
    (which model won, why, trade-offs considered)
    
    ## 7. Key Findings
    (3-5 most important insights from this research)
    
    ## 8. Recommendations
    (what to do next — what experiments to run, what data to collect)
    
    ## 9. Conclusion

    Be professional, specific with numbers, and write like a 
    senior data scientist presenting to hospital executives.
    """

    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    summary = result["messages"][-1].content
    return {"summary": summary}