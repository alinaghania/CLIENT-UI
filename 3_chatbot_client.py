from langchain_aws import BedrockLLM
import streamlit as st
import functools
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrock
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
import os
import boto3

# Configuration AWS with Streamlit secrets or environment variables
def aws_configure():
    """Configures AWS credentials and region from secrets or environment variables."""
    aws_access_key_id = st.secrets.get("aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key = st.secrets.get("aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY"))
    region_name = st.secrets.get("region_name", os.getenv("AWS_DEFAULT_REGION"))

    if not aws_access_key_id or not aws_secret_access_key or not region_name:
        st.error("AWS credentials or region not found!")
        return None

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    return session

# Call the AWS configuration function
session = aws_configure()
if session is None:
    st.stop()  # Stops execution if AWS credentials are missing

# Configuration de la page Streamlit
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

# Session state pour l'historique des chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    
def load_css(file_path):
    """Load CSS from a file."""
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Charger le CSS à partir du fichier styles.css
load_css("style.css")

# Function to measure execution time during streaming
def measure_time(func):
    @functools.wraps(func)
    def wrapper_measure_time(*args, **kwargs):
        start_time = datetime.now()
        first_token_time = None
        result = None

        def streaming_wrapper():
            nonlocal first_token_time, result
            for token in func(*args, **kwargs):
                if first_token_time is None:
                    first_token_time = datetime.now()
                yield token
            end_time = datetime.now()
            total_elapsed = (end_time - start_time).total_seconds()
            streaming_elapsed = (end_time - first_token_time).total_seconds() if first_token_time else 0

        return streaming_wrapper()

    return wrapper_measure_time

# Function to add a message to the chat history
def add_message_to_history(message):
    history = st.session_state.chat_history
    if len(history.messages) == 0 or type(history.messages[-1]) != type(message):
        history.add_message(message)

# Function to choose the model on Bedrock
def choose_model():
    """Chooses the Claude 3.5 Sonnet model from Amazon Bedrock."""
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm

# Function to initialize the chain with the model, prompt, and output parser
def initialize_chain():
    global system_prompt
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    bedrock_llm = choose_model()
    chain = prompt | bedrock_llm | StrOutputParser()

    wrapped_chain = RunnableWithMessageHistory(
        chain,
        lambda _: st.session_state.chat_history,  # Add chat history handling
        history_messages_key="chat_history",
    )

    return wrapped_chain

# Function to run the chain and apply session configuration
@measure_time
def run_chain(input_text, context):
    chain = initialize_chain()
    
    # Define session configuration
    config = {"configurable": {"session_id": "unique_session_id"}}
    
    # Stream response with session configuration
    response = chain.stream({"input": input_text, "context": context}, config)
    return response

# Load context if necessary
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

# User input and chain execution
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    add_message_to_history(HumanMessage(content=user_input))
    
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    with st.chat_message("AI"):
        response = run_chain(user_input, context)
        st.write(response)
    
    add_message_to_history(AIMessage(content=response))
