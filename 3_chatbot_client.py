from langchain_aws import BedrockLLM
import streamlit as st
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
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

# Setup your Bedrock credentials from Streamlit secrets
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Initialiser l'historique des messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fonction pour ajouter un message tout en respectant l'alternance
def add_message_to_history(message):
    history = st.session_state.chat_history
    if len(history) == 0 or type(history[-1]) != type(message):
        history.append(message)

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
        ("system", system_prompt),  # Le prompt système lu depuis le fichier
        ("chat_history", "{chat_history}"),  # Historique des messages pour maintenir le contexte
        ("human", "{input}")  # Le message de l'utilisateur
    ])

    # Obtenir le modèle choisi via la fonction choose_model()
    bedrock_llm = choose_model()

    # Création de la chaîne en utilisant le modèle, le prompt et un output parser
    chain = prompt | bedrock_llm | StrOutputParser()

    # Envelopper la chaîne avec l'historique des messages pour maintenir la continuité du dialogue
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        lambda: st.session_state.chat_history,  # Utiliser l'historique des messages stocké dans la session
        history_messages_key="chat_history",
    )

    return wrapped_chain

# Fonction pour exécuter la chaîne
def run_chain(input_text, context, session_id):
    chain = initialize_chain()
    if chain is None:
        raise ValueError("Initialized chain is not valid.")

    config = {
        "configurable": {
            "session_id": session_id
        }
    }
    
    # Assurer que le contexte est utilisé lors de l'appel
    response = chain.stream({"input": [input_text], "context": context}, config)

    # Ajouter la question de l'utilisateur et la réponse de l'IA à l'historique
    add_message_to_history(HumanMessage(content=input_text))
    add_message_to_history(AIMessage(content=response))

    return response

# Charger le contexte depuis un fichier
def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()
load_context()

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Afficher le message utilisateur
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    # Obtenir la réponse de l'IA et mesurer le temps
    with st.chat_message("AI"):
        response = run_chain(user_input, context, session_id="peugeot_expert")
        st.write(response)
