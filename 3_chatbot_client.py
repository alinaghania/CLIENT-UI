from langchain_aws import BedrockLLM
import streamlit as st
import functools
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
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

# Initialisation de l'historique des messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# Fonction pour mesurer le temps d'exécution pendant le streaming
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
 
@st.cache_data
def load_context():
    return Path("parsed_data/peugeot_data.txt").read_text()

# Chargement du contexte avec mise en cache
context = load_context()
 
 
 
 
@st.cache_resource
def choose_model():
    # Choix du modèle Claude 3.5 Sonnet depuis Amazon Bedrock
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm
 
@st.cache_resource
def initialize_chain():
    global system_prompt
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()
    
    # Charger le contexte dans le prompt système
    context = load_context()
    system_prompt = system_prompt.replace("{context}", context)
    
    # Define the prompt correctly
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    bedrock_llm = choose_model()
    
    # Ensure bedrock_llm is not None and is properly configured
    if bedrock_llm is None:
        raise ValueError("Bedrock model not initialized correctly.")
    
    # Create the chain
    chain = prompt | bedrock_llm | StrOutputParser()
    
    # Ensure chain is not None
    if chain is None:
        raise ValueError("Chain creation failed.")
    
    return chain
 
def run_chain(input_text, context, session_id):
    chain = initialize_chain()
    if chain is None:
        raise ValueError("Initialized chain is not valid.")
    
    config = {
        "configurable": {
            "session_id": session_id
        }
    }
    
    # Convertir l'historique en format approprié pour le modèle
    # chat_history_messages = [message.content for message in st.session_state.chat_history.messages]
    chat_history_messages = [str(message.content) for message in st.session_state.chat_history.messages]

    
    # # Ajouter l'historique des messages au contexte
    # full_input = f"{context}\n{' '.join(chat_history_messages)}\n{input_text}"
    
    full_input = {
        "input": input_text,
        "chat_history": " ".join(chat_history_messages)
    }
    
    # response = chain.stream({"input": [full_input], "context": context}, config)  # Wrap input_text in a list
    response = chain.stream(full_input, config)
    
    return response



# Fonction pour ajouter un message à l'historique
def add_message_to_history(message):
    st.session_state.chat_history.add_message(message)
 
# Afficher tout l'historique des messages avant d'ajouter la nouvelle interaction
for message in st.session_state.chat_history.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Ajouter le message utilisateur à l'historique
    add_message_to_history(HumanMessage(content=user_input))
    
    # Afficher le message utilisateur
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    # Afficher la réponse en flux
    with st.chat_message("AI"):
        # Création d'un conteneur Streamlit pour afficher la réponse de l'IA au fur et à mesure
        response_placeholder = st.empty()  # Crée un conteneur vide
        response_text = ""  # Chaîne pour stocker la réponse finale
        

        for token in run_chain(user_input, context, session_id="peugeot_expert"):
            response_text += token  # Ajouter chaque token à la réponse
            response_placeholder.markdown(response_text)  # Afficher le texte incrémental
        
        # Ajouter la réponse complète de l'IA à l'historique
        add_message_to_history(AIMessage(content=response_text))
