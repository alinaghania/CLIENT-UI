import os
from pathlib import Path

import boto3
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from utils import (
    check_question_type,
    initialize_chain_commercial,
    initialize_chain_experts_ev,
    initialize_chain_expert_data_ev_capacity
)

# Setup your Bedrock credentials from Streamlit secrets
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Streamlit page config
st.set_page_config(page_title="Peugeot Expert", page_icon="🚗")
st.title("EV - Peugeot Expert")

# Load CSS (optional)
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to display the chat history
def display_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(f"**You:** {message.content}")
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(f"**Peugeot Expert:** {message.content}")

# Display the chat history (for both user and AI)
display_chat_history()

# User input section
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Display the user message immediately
    with st.chat_message("Human"):
        st.markdown(f"**You:** {user_input}")

    # Format the history for context
    chat_history_messages = [msg.content for msg in st.session_state.chat_history]
    formatted_history = "\n".join(chat_history_messages)

    # Determine if the question is relevant to experts or commercial
    # relevance_result = check_question_type(user_input, formatted_history)

    # # Initialize the correct chain based on relevance

    # if relevance_result == "yes":
    #         chain = initialize_chain_experts_ev(formatted_history, user_input)
    #         print("chaine experts")
    # elif relevance_result == "ok":
    #     chain = initialize_chain_expert_data_ev_capacity(formatted_history, user_input)
    #     print("chaine expert data EV capacity")
    # elif relevance_result == "no":
    #     chain = initialize_chain_commercial(formatted_history, user_input)
    #     print("chaine commercial")
    # else:
    #     chain = initialize_chain_commercial(formatted_history, user_input)
    #     print("chaine commercial")
    

    # # Invoke the chain and get the output
    # result = chain.invoke({
    #     "user_input": user_input,
    #     "history": formatted_history,
    #     "context": ""  # Assuming context is handled within the chain initialization
    # })

    # # Stream the AI's response
    # with st.chat_message("AI"):
    #     response_placeholder = st.empty()
    #     response_text = result["response"]  # Extract the response
    #     key_words = result["key_words"]  # Extract the key words

    #     # Display the AI's response
    #     response_placeholder.markdown(f"**Peugeot Expert:** {response_text}")
        
    #     # Optionally display the key words if needed
    #     st.markdown(f"**Key Words:** {', '.join(key_words)}")

    # # Add the AI's response to the chat history
    # st.session_state.chat_history.append(AIMessage(content=response_text))
    # Determine if the question is relevant to experts or commercial
    relevance_result = check_question_type(user_input, formatted_history)

    # Initialize the correct chain based on relevance
    if relevance_result == "yes":
        chain = initialize_chain_experts_ev(formatted_history, user_input)
        print("chaine experts")
    elif relevance_result == "ok":
        chain = initialize_chain_expert_data_ev_capacity(formatted_history, user_input)
        print("chaine expert data EV capacity")
    elif relevance_result == "no":
        chain = initialize_chain_commercial(formatted_history, user_input)
        print("chaine commercial")
    else:
        chain = initialize_chain_commercial(formatted_history, user_input)
        print("chaine commercial par defaut")

    # Assurez-vous que chain est défini avant d'invoquer la chaîne
    if chain is not None:
        result = chain.invoke({
            "user_input": user_input,
            "history": formatted_history,
            "context": ""  # Assuming context is handled within the chain initialization
        })

        # Stream the AI's response
        with st.chat_message("AI"):
            response_placeholder = st.empty()
            response_text = result["response"]  # Extract the response
            key_words = result["key_words"]  # Extract the key words

            # Display the AI's response
            response_placeholder.markdown(f"**Peugeot Expert:** {response_text}")
            
            # Optionally display the key words if needed
            st.markdown(f"**Key Words:** {', '.join(key_words)}")

        # Add the AI's response to the chat history
        st.session_state.chat_history.append(AIMessage(content=response_text))
    else:
        st.error("Erreur: Chaîne non initialisée correctement.")


