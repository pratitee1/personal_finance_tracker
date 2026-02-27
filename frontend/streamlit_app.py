import streamlit as st
import requests
import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# Config
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="Personal Finance Tracker", layout="wide")

st.title("💰 Personal Finance Tracker")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page:", ["Chat", "Upload Receipts"])

# Chat Interface
if page == "Chat":
    st.header("💬 Ask Questions About Your Finances")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar filters for chat
    with st.sidebar:
        st.subheader("Query Filters")
        user_id = st.number_input("User ID", value=1, min_value=1, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2024, 1, 1))
        with col2:
            end_date = st.date_input("End Date")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your finances..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Call RAG/question pipeline
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = requests.post(
                        f"{API_BASE_URL}/rag/Question",
                        json={
                            "question": prompt,
                            "user_id": user_id,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat()
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    answer = response.json().get("answer", "No response from API")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")

# Upload Receipts Interface
elif page == "Upload Receipts":
    st.header("📸 Upload Receipts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        image_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if image_file and st.button("Upload Image Receipt"):
            try:
                files = {"file": image_file}
                with st.spinner("Processing image..."):
                    response = requests.post(
                        f"{API_BASE_URL}/upload/receipt",  # adjust endpoint as needed
                        files=files,
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success("✅ Receipt uploaded successfully!")
                    st.json(result)
            except requests.exceptions.RequestException as e:
                st.error(f"Upload Error: {str(e)}")
    
    with col2:
        st.subheader("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
        if pdf_file and st.button("Upload PDF Receipt"):
            try:
                files = {"file": pdf_file}
                with st.spinner("Processing PDF..."):
                    response = requests.post(
                        f"{API_BASE_URL}/upload/receipt",  # adjust endpoint as needed
                        files=files,
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success("✅ Receipt uploaded successfully!")
                    st.json(result)
            except requests.exceptions.RequestException as e:
                st.error(f"Upload Error: {str(e)}")

st.markdown("---")
st.caption(f"Connected to: {API_BASE_URL}")