import streamlit as st
import yaml
import os
import re

# Function to load credentials from the YAML file
def load_credentials():
    DATA_FOLDER = os.path.join(os.getcwd(), "hotel_ui", "data", "login")
    assert os.path.exists(DATA_FOLDER), "Data folder does not exist"

    with open(os.path.join(DATA_FOLDER, 'credentials.yaml'), 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials['users']

# Function to verify the credentials
def verify_credentials(username, password):
    users = load_credentials()
    for user in users:
        if user['username'] == username and user['password'] == password:
            return True, user['username']
    return False, None

# Function to extract user ID from username
def extract_user_id(username):
    match = re.search(r'hotel_(\w+)', username)
    if match:
        return match.group(1)
    return None

# Main function to display the login form
def show_login():
    # Apply a custom theme for the login page
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to right, #00b4db, #0083b0);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the login form
    with st.container():
        st.markdown("<h1 style='text-align: center; color: green;'>Welcome team Pullman!</h1>", unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username", key="username")
            password = st.text_input("Password", type="password", key="password")

            submit_button = st.form_submit_button("Login")

            if submit_button:
                success, user_id = verify_credentials(username, password)
                if success:
                    st.session_state['user'] = username
                    st.experimental_rerun()  # Rerun the app to refresh after login
                else:
                    st.error("Invalid username or password")
