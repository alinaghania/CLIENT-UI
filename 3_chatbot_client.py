import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path
import os
from datetime import datetime

# Setup AWS credentials from Streamlit secrets
os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Streamlit interface configuration
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

# Load custom CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("style.css")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = RunnableWithMessageHistory()

# Define the Bedrock model
def choose_model():
    return ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Update the model if needed
        model_kwargs={"temperature": 0.1}  # Adjust parameters as required
    )

# Function to initialize the prompt chain
def initialize_chain():
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    bedrock_llm = choose_model()
    chain = prompt | bedrock_llm
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        lambda: st.session_state.chat_history,  # Attach chat history
        history_messages_key="chat_history"
    )
    return wrapped_chain

# Time measurement function
def measure_time(func):
    def wrapper_measure_time(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        total_elapsed = (end_time - start_time).total_seconds()
        st.write(f"Total response time: {total_elapsed:.2f} seconds.")
        return result
    return wrapper_measure_time

# Run chain function
@measure_time
def run_chain(input_text, context):
    chain = initialize_chain()
    config = {"configurable": {"session_id": "unique_session_id"}}
    response = chain.stream({"input": input_text, "context": context}, config)
    return response

# Load context data
context = None
def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()
load_context()

# Display chat history
for message in st.session_state.chat_history.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Add user message to history
    st.session_state.chat_history.add_message(HumanMessage(content=user_input))
    
    # Display user input
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    # Get AI response
    with st.chat_message("AI"):
        response = run_chain(user_input, context)
        st.write(response)
    
    # Add AI response to history
    st.session_state.chat_history.add_message(AIMessage(content=response))

