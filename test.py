import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
import os

# Setup your Bedrock credentials from Streamlit secrets
import boto3
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Streamlit interface
st.set_page_config(page_title="Bedrock Chat")
st.title("Simple Bedrock Chatbot")

# Initialize the Bedrock Chat model (Claude 3)
chat = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240620-v1:0",  # You can change the model as per your requirements
    session=session,  # Use the boto3 session
    model_kwargs={"temperature": 0.1}  # Adjust parameters as needed
)

# Input from the user
user_input = st.text_input("Ask anything:", placeholder="Enter your question here...")

# If the user provides input, send it to Bedrock for processing
if user_input:
    # Prepare the message in the correct format
    messages = [HumanMessage(content=user_input)]
    
    # Stream the response from Bedrock
    for chunk in chat.stream(messages):
        st.write(chunk.content)
