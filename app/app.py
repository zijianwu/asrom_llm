import hashlib
import time

import requests
import streamlit as st

QA_API_ENDPOINT = "http://localhost:5000/qa"
CLINICAL_REFERENCE_API_ENDPOINT = "http://localhost:5000/clinical_reference"
TEST_USER_NAME = "user"
TEST_PASSWORD = "password"


def authenticate(username, password):
    # Your authentication logic goes here
    # Here's an example using a hardcoded username and password for demonstration purposes
    if username == TEST_USER_NAME:
        salt = "RANDOM_STRING"  # Replace with a secure random salt
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        expected_hash = hashlib.sha256(
            (TEST_PASSWORD + salt).encode()
        ).hexdigest()  # Replace with a hashed password for the user
        if password_hash == expected_hash:
            return True
    else:  # TODO: Create option to look up usernames/passwords in separate database
        pass
    return False


def requires_authentication(func):
    def wrapper(*args, **kwargs):
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
        if not st.session_state["logged_in"]:
            st.warning("Please log in to use this app.")
            return
        return func(*args, **kwargs)

    return wrapper


@requires_authentication
def question_and_answer_page():
    query = st.text_input("Enter question")
    if st.button("Search"):
        with st.spinner("Loading..."):
            response = requests.get(QA_API_ENDPOINT, params={"query": query})
        st.write(response.text)


@requires_authentication
def clinical_reference_page():
    # TODO: Add a table of contents to the UI
    # TODO: Allow users to flag specific sections of the document (e.g., correctness)
    # TODO: Allow users to rate articles
    # TODO: Allow users to make edits
    query = st.text_input("Enter title for clinical reference article")
    if st.button("Search"):
        with st.spinner("Loading..."):
            response = requests.get(CLINICAL_REFERENCE_API_ENDPOINT, params={"query": query})
        st.markdown(response.text)


def login_form():
    st.write("# Please log in to use this app")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            with st.spinner("Redirecting to application..."):
                time.sleep(1)
                st.experimental_rerun()


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if not st.session_state.logged_in:
    login_form()
else:
    search_version = st.sidebar.selectbox(
        "Choose functionality", ("Clinical Reference", "Question and Answer")
    )
    if search_version == "Question and Answer":
        question_and_answer_page()
    elif search_version == "Clinical Reference":
        clinical_reference_page()
