# streamlit_app.py
import streamlit as st
from rag_engine import query_chatbot

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📊 Financial Advisor RAG Chatbot")
st.markdown("Ask questions about Fixed Deposits based on our pre-loaded document.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask your question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    response = query_chatbot(prompt, st.session_state.chat_history)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
