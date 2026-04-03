# agents/data_explorer.py
# This is the Data Explorer Agent.
# It uses LangGraph's create_react_agent — the simplest 
# agent pattern. Give it an LLM + tools and it figures 
# out what to call and when.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.data_tools import (
    profile_dataset,
    check_missing_values,
    analyze_numeric_features,
    check_class_balance,
    detect_outliers
)

load_dotenv()


def run_data_explorer(
    file_path: str,
    target_column: str,
    problem_statement: str
) -> str:
    """
    Run the Data Explorer Agent on a dataset.
    
    Args:
        file_path: path to the CSV file
        target_column: the column we want to predict
        problem_statement: plain English description of the task
    
    Returns:
        A full EDA report as a string
    """

    # The LLM that powers our agent's reasoning
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # 0 = deterministic, we want consistent analysis
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # All tools this agent is allowed to use
    tools = [
        profile_dataset,
        check_missing_values,
        analyze_numeric_features,
        check_class_balance,
        detect_outliers
    ]

    # Create the agent — this is all LangGraph needs
    agent = create_react_agent(llm, tools)

    # The task we give the agent
    prompt = f"""
    You are a Data Explorer Agent — a senior data scientist doing 
    exploratory data analysis.

    Dataset location: {file_path}
    Target variable: {target_column}
    Problem: {problem_statement}

    Use ALL available tools to fully analyze this dataset:
    1. Profile the dataset
    2. Check for missing values
    3. Analyze numeric features
    4. Check class balance for the target variable
    5. Detect outliers

    After using all tools, write a comprehensive EDA Report with:
    - Dataset Overview (rows, columns, types)
    - Data Quality Issues (missing values, outliers)
    - Class Balance Analysis
    - Key Feature Insights
    - Specific Recommendations for the Feature Engineering step
    
    Be specific with numbers. This report will be used by the 
    next agent in the pipeline.
    """

    # Run the agent
    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    # The last message is always the agent's final answer
    return result["messages"][-1].content