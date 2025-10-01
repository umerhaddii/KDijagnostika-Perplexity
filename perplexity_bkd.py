# new backend of agentic system for perplexity - Date: 30/09/2025 - 9:15 PM

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
from openai import OpenAI
import streamlit as st

SONAR_PROMPT = """You are an automotive diagnostic search assistant.

Rules:
1. Provide only clean technical information without any citation numbers or reference brackets.
2. Do not include [1], [2], [3] or any similar reference markers in your response.
3. Present information in clear, readable paragraphs.

Task:
Search the web for current technical information about automotive diagnostics including:
- Detailed troubleshooting information
- Root causes and symptoms
- Technical solutions and specifications
- Step-by-step diagnostic procedures

Format:
- Write in natural paragraphs
- Be comprehensive and technical
- Remove all citation markers from your response"""

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

