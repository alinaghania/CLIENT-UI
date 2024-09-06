from langchain.memory import ConversationBufferMemory
from langchain_aws import BedrockLLM
import streamlit as st
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from pathlib import Path
import boto3
import os

# Configuration de la page Streamlit
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Charger le CSS à partir du fichier styles.css
load_css("style.css")

# Setup Bedrock credentials
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Fonction pour choisir le modèle sur Bedrock
def choose_model():
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm

# Fonction pour initialiser la chaîne avec mémoire
def initialize_chain():
    global system_prompt
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{input}")
    ])
    
    bedrock_llm = choose_model()

    # Initialize memory
    memory = ConversationBufferMemory(return_messages=True)
    
    # Create ConversationChain with memory
    chain = ConversationChain(llm=bedrock_llm, memory=memory, prompt=prompt)

    return chain

# Fonction pour exécuter la chaîne
def run_chain(input_text, context):
    chain = initialize_chain()
    response = chain.predict(input_text=input_text)
    return response

# Load context
context = None
def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()
load_context()

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    with st.chat_message("AI"):
        response = run_chain(user_input, context)
        st.write(response)
