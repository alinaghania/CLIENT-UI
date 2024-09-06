from langchain.memory import ConversationBufferMemory  # Import memory
from langchain_aws import BedrockLLM
import streamlit as st
import functools
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
import boto3
import os

# Configuration de la page Streamlit
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

# Charger le CSS à partir du fichier styles.css
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("style.css")

# Setup your Bedrock credentials from Streamlit secrets
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Initialize memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to measure execution time during streaming
def measure_time(func):
    @functools.wraps(func)
    def wrapper_measure_time(*args, **kwargs):
        start_time = datetime.now()
        first_token_time = None
        
        def streaming_wrapper():
            nonlocal first_token_time
            for token in func(*args, **kwargs):
                if first_token_time is None:
                    first_token_time = datetime.now()
                yield token
            end_time = datetime.now()
            total_elapsed = (end_time - start_time).total_seconds()
            streaming_elapsed = (end_time - first_token_time).total_seconds() if first_token_time else 0
            # Optionally output time metrics
            # st.write(f" Total response time: {total_elapsed:.2f} seconds.")
            # st.write(f"Streaming time: {streaming_elapsed:.2f} seconds.")
        
        return streaming_wrapper()
    
    return wrapper_measure_time

# Choose the model on Bedrock
def choose_model():
    # Choose Claude 3.5 Sonnet from Amazon Bedrock
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm

# Initialize the chain with the Bedrock model and prompt
def initialize_chain():
    global system_prompt
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()
    
    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{input}")
    ])
    
    bedrock_llm = choose_model()
    
    # Ensure the Bedrock model is properly configured
    if bedrock_llm is None:
        raise ValueError("Bedrock model not initialized correctly.")
    
    # Create the chain (Note: memory is not directly part of the RunnableSequence, so manage memory outside)
    chain = prompt | bedrock_llm | StrOutputParser()
    
    return chain

@measure_time
def run_chain(input_text, context, session_id):
    chain = initialize_chain()
    
    # Retrieve conversation history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    config = {
        "configurable": {
            "session_id": session_id
        }
    }
    
    # Run the chain with both the user's input and the chat history
    response = chain.stream({"input": [input_text], "context": context, "chat_history": chat_history}, config)
    
    # Update memory with new context
    memory.save_context({"input": input_text}, {"response": response})
    
    return response

context = None

# Load context
def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()
load_context()

# User input handling
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Display the user's message
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    # Get AI response and display it
    with st.chat_message("AI"):
        response = run_chain(user_input, context, session_id="peugeot_expert")
        st.write(response)
        