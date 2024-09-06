import streamlit as st
from langchain_aws import BedrockLLM
import boto3

# Set up AWS credentials
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

# Create Bedrock client and model setup
bedrock_runtime = session.client('bedrock-runtime')
llm = BedrockLLM(client=bedrock_runtime, model_id="anthropic.claude-1")

# Streamlit UI Setup
st.title("Simple Chatbot")
st.write("Ask anything and get a response from the Claude LLM")

# Chat input
user_input = st.text_input("Your question:", "")

# If user input is given
if user_input:
    response = llm.invoke_model(input_text=user_input)
    st.write(f"Response: {response['body']['output']}")
