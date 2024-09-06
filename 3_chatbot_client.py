from langchain_aws import BedrockLLM
import streamlit as st
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import boto3
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory


# Configuration de la page Streamlit
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Charger le CSS à partir du fichier styles.css
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

# Initialiser l'historique des messages avec InMemoryChatMessageHistory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# Fonction pour ajouter un message tout en respectant l'alternance
def add_message_to_history(message):
    st.session_state.chat_history.add_message(message)

# Fonction pour choisir le modèle sur Bedrock
def choose_model():
    # Choix du modèle Claude 3.5 Sonnet depuis Amazon Bedrock
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm

# Fonction d'initialisation de la chaîne
def initialize_chain():
    # Lire le prompt système depuis le fichier externe "prompt/system_prompt.txt"
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()

    # Définir le template du prompt avec les messages pour le système et l'utilisateur
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),  # Le prompt système lu depuis le fichier
        MessagesPlaceholder(variable_name="chat_history"),  # Historique des messages pour maintenir le contexte
        HumanMessage(content="{input}")  # Le message de l'utilisateur
    ])

    # Obtenir le modèle choisi via la fonction choose_model()
    bedrock_llm = choose_model()

    # Création de la chaîne en utilisant le modèle, le prompt et un output parser
    chain = prompt | bedrock_llm | StrOutputParser()

    # Envelopper la chaîne avec l'historique des messages pour maintenir la continuité du dialogue
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        lambda session: st.session_state.chat_history.messages,  # Accept the session and get the chat history
        history_messages_key="chat_history",
    )

    return wrapped_chain


def run_chain(input_text, context, session_id):
    chain = initialize_chain()
    if chain is None:
        raise ValueError("Initialized chain is not valid.")

    config = {
        "configurable": {
            "session_id": session_id
        }
    }

    # Print the config to debug
    print(f"Config: {config}")

    # Try streaming the response
    response_container = st.empty()
    response_text = ""

    try:
        for token in chain.stream({"input": [input_text], "context": context}, config):
            response_text += token
            response_container.markdown(response_text)  # Update the response incrementally
    except Exception as e:
        st.error(f"Error during streaming: {e}")
        print(f"Error: {e}")

    return response_text


context = None
# Charger le contexte depuis un fichier
def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()
load_context()

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Ajouter le message utilisateur à l'historique
    add_message_to_history(HumanMessage(content=user_input))
    
    # Afficher le message utilisateur
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    # Obtenir la réponse de l'IA et l'ajouter à l'historique
    with st.chat_message("AI"):
        response = run_chain(user_input, context, session_id="peugeot_expert")
        add_message_to_history(AIMessage(content=response))
