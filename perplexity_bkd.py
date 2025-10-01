# new backend of agentic system for perplexity - Date: 30/09/2025 - 9:15 PM

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
from openai import OpenAI
import streamlit as st

SONAR_PROMPT = """Search the web for current technical information about automotive diagnostics. 
Find detailed troubleshooting information, causes, solutions, and technical specifications.
Include comprehensive technical data that will help diagnose automotive issues.

IMPORTANT: Provide only the information without any citation numbers, brackets, or reference links. Give clean text responses only."""

# State definition
class MainState(TypedDict):
    user_question: str
    sonar_response: str

# Initialize Perplexity model
def initialize_perplexity():
    perplexity_key = st.secrets["PERPLEXITY_API_KEY"]
    perplexity_client = OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai/")
    return perplexity_client

# Node function
def sonar_search_node(state: MainState) -> MainState:
    perplexity_client = initialize_perplexity()
    
    combined_prompt = f"{SONAR_PROMPT}\n\nUser question: {state['user_question']}"
    
    response = perplexity_client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0,
        max_tokens=1500
    )
    
    state["sonar_response"] = response.choices[0].message.content
    return state

# Build LangGraph workflow
def create_workflow():
    checkpoint = InMemorySaver()
    
    graph = StateGraph(MainState)
    graph.add_node("sonar_search", sonar_search_node)
    
    graph.add_edge(START, "sonar_search")
    graph.add_edge("sonar_search", END)
    
    workflow = graph.compile(checkpointer=checkpoint)
    return workflow

# Streaming workflow execution
def stream_diagnostic_workflow(user_question, thread_id="default"):
    workflow = create_workflow()
    
    initial_state = {
        "user_question": user_question,
        "sonar_response": ""
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    for event in workflow.stream(initial_state, config):
        node_name = list(event.keys())[0]
        node_output = event[node_name]

        yield node_name, node_output
