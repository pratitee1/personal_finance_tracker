import streamlit as st

st.set_page_config(
    page_title="Finance App",
    layout="wide",
)

st.sidebar.title("Navigation")
tabs = ["Overview", "Trends", "Alerts", "Review & Correct", "Chat"]
selection = st.sidebar.radio("Go to", tabs)

st.write("Hello Finance App")
