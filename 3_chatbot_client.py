import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
import os
from datetime import datetime
import functools
from pathlib import Path

# Set AWS environment variables from Streamlit secrets
os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Configuration of the Streamlit page
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write("<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# Load custom CSS for the app
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Function to measure execution time
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
            st.write(f"Total response time: {total_elapsed:.2f} seconds.")
            st.write(f"Streaming time: {streaming_elapsed:.2f} seconds.")

        return streaming_wrapper()

    return wrapper_measure_time

# Function to add messages to chat history
def add_message_to_history(message):
    history = st.session_state.chat_history
    if len(history.messages) == 0 or type(history.messages[-1]) != type(message):
        history.add_message(message)

# Function to choose the Bedrock model
def choose_model():
    return ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")

# Initialize the prompt and model chain
def initialize_chain():
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # Choose the model
    bedrock_llm = choose_model()

    # Create a chain using the model, prompt, and output parser
    chain = prompt | bedrock_llm | StrOutputParser()

    # Wrap the chain with message history to maintain dialogue continuity
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        lambda: st.session_state.chat_history,  # Function to get chat history
        history_messages_key="chat_history"
    )

    return wrapped_chain

# Function to run the model chain
@measure_time
def run_chain(input_text, context):
    # Initialize the chain
    chain = initialize_chain()

    # Configuration with session_id
    config = {"configurable": {"session_id": "unique_session_id"}}

    # Get the response by streaming
    response = chain.stream({"input": input_text, "context": context}, config)
    return response

# Load the context from a file (e.g., specific to Peugeot data)
context = None
def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()

load_context()

# Display the chat history
for message in st.session_state.chat_history.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input field
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Add the user input to chat history
    add_message_to_history(HumanMessage(content=user_input))

    # Display the user input
    with st.chat_message("Human"):
        st.markdown(user_input)

    # Get AI response and measure the time
    with st.chat_message("AI"):
        response = run_chain(user_input, context)
        st.write(response)

    # Add the AI response to chat history
    add_message_to_history(AIMessage(content=response))
