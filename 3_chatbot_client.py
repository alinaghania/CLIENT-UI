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

# Configuration des variables d'environnement AWS
os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]

# Configuration de la page Streamlit
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

# Initialiser la session AWS avec boto3
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

# Initialiser l'historique des chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Charger le CSS à partir du fichier styles.css
load_css("style.css")

# Fonction pour mesurer le temps d'exécution pendant le streaming
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

# Fonction pour vérifier et ajouter les messages tout en respectant l'alternance
def add_message_to_history(message):
    history = st.session_state.chat_history
    if len(history.messages) == 0:
        history.add_message(message)
    else:
        # Assurer l'alternance des rôles
        last_message_role = type(history.messages[-1]).__name__.lower()
        current_message_role = type(message).__name__.lower()
        if last_message_role != current_message_role:
            history.add_message(message)
        else:
            st.write(f"Message non ajouté: le rôle '{current_message_role}' est identique à celui du dernier message.")

# Fonction pour choisir le modèle sur Bedrock
def choose_model():
    # Choix du modèle Claude 3.5 Sonnet depuis Amazon Bedrock
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm

# Fonction pour créer et initialiser la chaîne
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
        lambda _: st.session_state.chat_history,
        history_messages_key="chat_history",
    )

    return wrapped_chain

# Appliquer le décorateur pour mesurer le temps d'exécution
@measure_time
def run_chain(input_text, context):
    chain = initialize_chain()
    config = {"configurable": {"session_id": "unique_session_id"}}
    response = chain.stream({"input": input_text, "context": context}, config)
    return response

# Charger le contexte du document
context = None

def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()

load_context()

# Afficher l'historique des messages
for message in st.session_state.chat_history.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Ajouter le message utilisateur à l'historique
    add_message_to_history(HumanMessage(content=user_input))
    
    # Afficher le message utilisateur
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    # Obtenir la réponse de l'IA et mesurer le temps
    with st.chat_message("AI"):
        response = run_chain(user_input, context)
        st.write(response)
    
    # Ajouter la réponse de l'IA à l'historique
    add_message_to_history(AIMessage(content=response))
