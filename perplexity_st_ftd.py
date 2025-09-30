# new frontend of agentic system for perplexity - Date: 30/09/2025 - 9:15 PM 
# the backend associated with this file is perplexity_bkd.py

import streamlit as st
import uuid
from perplexity_bkd import stream_diagnostic_workflow

st.title("KDijagnostika Automotive Diagnostic Support")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar with New Chat button
with st.sidebar:
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.subheader("🌐 Diagnostic Response:")
            st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# Chat input
user_question = st.chat_input("Enter your automotive diagnostic question:")

if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Assistant response
    with st.chat_message("assistant"):
        sonar_response = ""
        
        # Initialize status container
        status_container = st.status("🔍 Searching web for diagnostic information...")
        
        # Stream through LangGraph workflow
        for node_name, node_output in stream_diagnostic_workflow(user_question, st.session_state.thread_id):
            if node_name == "sonar_search":
                sonar_response = node_output["sonar_response"]
                status_container.update(label="✅ Search completed", state="complete")
                
                st.subheader("🌐 Diagnostic Response:")
                st.markdown(sonar_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": sonar_response
    })