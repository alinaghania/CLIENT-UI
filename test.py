from boto3 import Session
from botocore.client import Config
import streamlit as st
import json

# Configure your AWS session with Streamlit secrets
session = Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

bedrock_client = session.client("bedrock-runtime", config=Config())

def invoke_model(prompt, model_id):
    # Create payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }

    # Invoke the model
    response = bedrock_client.invoke_model(
        ModelId=model_id,
        Body=json.dumps(payload),
        ContentType="application/json"
    )

    # Return the response content
    return json.loads(response["Body"].read())["completions"][0]["text"]

# Usage
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
prompt = "Describe the purpose of a 'hello world' program."
response = invoke_model(prompt, model_id)
st.write(response)
