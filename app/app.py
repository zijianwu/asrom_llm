import hashlib
import time

import requests
import streamlit as st

API_ENDPOINT = "http://localhost:5000/search"
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
def search():
    search_query = st.text_input("Enter Search Query")
    if st.button("Search"):
        with st.spinner("Loading..."):
            response = requests.get(
                API_ENDPOINT, params={"query": search_query}
            )
        st.write(response.text)


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
    search()
