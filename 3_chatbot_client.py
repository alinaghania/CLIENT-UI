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
# Configuration de la page Streamlit
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)
# Session state pour l'historique des chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    
import boto3
import os

# Setup your Bedrock credentials from Streamlit secrets
session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["region_name"]
)

os.environ["AWS_DEFAULT_REGION"] = st.secrets["region_name"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws_secret_access_key"]
    
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
                    # st.write(f"<span style='color:red;font-weight:bold'>First token received in {(first_token_time - start_time).total_seconds():.2f} seconds.</span>", unsafe_allow_html=True)
                yield token
            end_time = datetime.now()
            total_elapsed = (end_time - start_time).total_seconds()
            streaming_elapsed = (end_time - first_token_time).total_seconds() if first_token_time else 0
            # st.write(f" Total response time: {total_elapsed:.2f} seconds.")
            # st.write(f"Streaming time: {streaming_elapsed:.2f} seconds.")
        
        return streaming_wrapper()
    
    return wrapper_measure_time
# Fonction pour vérifier et ajouter les messages tout en respectant l'alternance
def add_message_to_history(message):
    history = st.session_state.chat_history
    if len(history.messages) == 0 or type(history.messages[-1]) != type(message):
        history.add_message(message)
        

# Fonction pour choisir le modèle sur Bedrock
def choose_model():
    # Choix du modèle Claude 3.5 Sonnet depuis Amazon Bedrock
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm




def initialize_chain():
    # Lire le prompt système depuis le fichier externe "prompt/system_prompt.txt"
    global system_prompt
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()
    # Définir le template du prompt avec les messages pour le système et l'utilisateur
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # Le prompt système lu depuis le fichier
        ("placeholder", "{chat_history}"),  # Historique des messages pour maintenir le contexte
        ("human", "{input}")  # Le message de l'utilisateur
    ])
    # Obtenir le modèle choisi via la fonction choose_model()
    bedrock_llm = choose_model()
    # Création de la chaîne en utilisant le modèle, le prompt et un output parser
    chain = prompt | bedrock_llm | StrOutputParser()
    # Envelopper la chaîne avec l'historique des messages pour maintenir la continuité du dialogue
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        lambda: st.session_state.chat_history,
        history_messages_key="chat_history",
    )
    return wrapped_chain




# Appliquer le décorateur pour mesurer le temps d'exécution
@measure_time
def run_chain(input_text, context):
    chain = initialize_chain()
    response = chain.stream({"input": input_text, "context": context})
    return response
# # Charger le contexte du document
# context = Path("parsed_data/peugeot_data.txt").read_text()
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
    
    
