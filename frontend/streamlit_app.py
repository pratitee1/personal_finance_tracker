import os
from datetime import date

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60

st.set_page_config(page_title="Personal Finance Tracker", layout="wide")


def initialize_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("current_user", None)


def auth_headers():
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"Request failed with status {response.status_code}"
    if isinstance(payload, dict):
        return payload.get("detail") or str(payload)
    return str(payload)


def fetch_current_user() -> bool:
    token = st.session_state.get("access_token")
    if not token:
        st.session_state.current_user = None
        return False

    try:
        response = requests.get(
            f"{API_BASE_URL}/auth/me",
            headers=auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        st.session_state.access_token = None
        st.session_state.current_user = None
        return False

    if response.ok:
        st.session_state.current_user = response.json()
        return True

    st.session_state.access_token = None
    st.session_state.current_user = None
    return False


def register_user(name: str, email: str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/auth/register",
        json={"name": name, "email": email, "password": password},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(api_error_message(response))


def login_user(email: str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/auth/token",
        data={"username": email, "password": password},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(api_error_message(response))

    token = response.json()["access_token"]
    st.session_state.access_token = token
    if not fetch_current_user():
        raise RuntimeError("Login succeeded but user profile lookup failed.")


def logout_user():
    st.session_state.access_token = None
    st.session_state.current_user = None
    st.session_state.messages = []


def render_auth_sidebar():
    st.sidebar.header("Authentication")
    current_user = st.session_state.get("current_user")

    if current_user:
        st.sidebar.success(f"Signed in as {current_user['email']}")
        st.sidebar.caption(f"User ID: {current_user['id']}")
        if st.sidebar.button("Log Out", use_container_width=True):
            logout_user()
            st.rerun()
        return

    auth_tab_login, auth_tab_register = st.sidebar.tabs(["Login", "Register"])

    with auth_tab_login:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Log In", use_container_width=True)
        if login_submitted:
            try:
                login_user(login_email, login_password)
            except (requests.RequestException, RuntimeError) as exc:
                st.sidebar.error(f"Login failed: {exc}")
            else:
                st.sidebar.success("Logged in.")
                st.rerun()

    with auth_tab_register:
        with st.form("register_form"):
            register_name = st.text_input("Name", key="register_name")
            register_email = st.text_input("Email", key="register_email")
            register_password = st.text_input(
                "Password",
                type="password",
                key="register_password",
            )
            register_submitted = st.form_submit_button("Create Account", use_container_width=True)
        if register_submitted:
            try:
                register_user(register_name, register_email, register_password)
                login_user(register_email, register_password)
            except (requests.RequestException, RuntimeError) as exc:
                st.sidebar.error(f"Registration failed: {exc}")
            else:
                st.sidebar.success("Account created.")
                st.rerun()


def ensure_authenticated() -> bool:
    if st.session_state.get("current_user"):
        return True
    st.info("Log in from the sidebar to upload receipts and use chat.")
    return False


def render_chat_page():
    st.header("Ask Questions About Your Finances")
    if not ensure_authenticated():
        return

    current_user = st.session_state["current_user"]
    st.caption(f"Authenticated as {current_user['email']}")

    with st.sidebar:
        st.subheader("Query Filters")
        start_date = st.date_input("Start Date", value=date(2025, 1, 1))
        end_date = st.date_input("End Date", value=date.today())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me anything about your finances...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_BASE_URL}/rag/Question",
                    json={
                        "question": prompt,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                    headers=auth_headers(),
                    timeout=REQUEST_TIMEOUT,
                )
                if not response.ok:
                    raise RuntimeError(api_error_message(response))
                answer = response.json().get("answer", "No response from API")
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"API error: {exc}")


def render_upload_page():
    st.header("Upload Receipts")
    if not ensure_authenticated():
        return

    st.caption("The current API accepts image uploads only.")
    image_file = st.file_uploader(
        "Choose a receipt image",
        type=["jpg", "jpeg", "png", "tiff"],
    )
    if not image_file:
        return

    if st.button("Upload Receipt", use_container_width=True):
        try:
            with st.spinner("Processing image..."):
                response = requests.post(
                    f"{API_BASE_URL}/upload/receipt",
                    files={"file": (image_file.name, image_file.getvalue(), image_file.type)},
                    headers=auth_headers(),
                    timeout=REQUEST_TIMEOUT,
                )
                if not response.ok:
                    raise RuntimeError(api_error_message(response))
                result = response.json()
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Upload failed: {exc}")
            return

        st.success("Receipt uploaded successfully.")
        st.json(result)


initialize_state()
if st.session_state.get("access_token") and not st.session_state.get("current_user"):
    fetch_current_user()

st.title("Personal Finance Tracker")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page:", ["Chat", "Upload Receipts"])

render_auth_sidebar()

if page == "Chat":
    render_chat_page()
else:
    render_upload_page()

st.markdown("---")
st.caption(f"Connected to: {API_BASE_URL}")
