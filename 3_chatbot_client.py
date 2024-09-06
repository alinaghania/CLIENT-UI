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
import boto3
import os

# Configuration de la page Streamlit
st.set_page_config(page_title="Peugeot Expert")
st.title("EV - Peugeot Expert")
st.write(f"<span style='color:red;font-weight:bold'> Expert en véhicules électriques Peugeot </span>", unsafe_allow_html=True)

# Session state pour l'historique des chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
else:
    if not isinstance(st.session_state.chat_history, InMemoryChatMessageHistory):
        st.session_state.chat_history = InMemoryChatMessageHistory()

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

def add_message_to_history(message):
    history = st.session_state.chat_history
    
    if not isinstance(history, InMemoryChatMessageHistory):
        print("Error: chat_history is not of type InMemoryChatMessageHistory.")
        return
    
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        raise ValueError("Invalid message type")
    
    # Ensure the last message role is not the same as the new one
    if len(history.messages) == 0 or isinstance(history.messages[-1], AIMessage) and role == "user":
        history.add_message(message)
    elif isinstance(history.messages[-1], HumanMessage) and role == "assistant":
        history.add_message(message)

# Fonction pour choisir le modèle sur Bedrock
def choose_model():
    # Choix du modèle Claude 3.5 Sonnet depuis Amazon Bedrock
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm

def initialize_chain():
    global system_prompt
    system_prompt_path = Path("prompt/system_prompt.txt")
    system_prompt = system_prompt_path.read_text()
    
    # Define the prompt correctly
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
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
    
    # Wrap the chain with message history
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        lambda _: get_chat_history(),
        history_messages_key="chat_history",
    )
    
    return wrapped_chain

def get_chat_history():
    # Retrieve chat history
    history = st.session_state.get('chat_history', InMemoryChatMessageHistory())
    
    # Check if history is an instance of InMemoryChatMessageHistory
    if not isinstance(history, InMemoryChatMessageHistory):
        print("Error: chat_history is unexpectedly of type", type(history))
        return []

    # Convert the history to a list of dictionaries for debugging
    history_list = []
    
    # Debug print
    print("Chat History:", history.messages)  # Inspect the messages

    # Check the type of each message
    for message in history.messages:
        if isinstance(message, (AIMessage, HumanMessage)):
            role = "assistant" if isinstance(message, AIMessage) else "user"
            content = message.content
            
            # Append to history list for further processing
            history_list.append({"role": role, "content": content})
        else:
            print(f"Error: Unexpected message type: {message}")

    # Validate the format of history items
    last_role = None
    for message in history_list:
        if last_role and last_role == message["role"]:
            print(f"Error: Consecutive roles found: {last_role} followed by {message['role']}")
        last_role = message["role"]
    
    return history_list

@measure_time
def run_chain(input_text, context, session_id):
    chain = initialize_chain()
    if not isinstance(chain, RunnableWithMessageHistory):
        raise ValueError("Initialized chain is not of type RunnableWithMessageHistory.")
    
    config = {
        "configurable": {
            "session_id": session_id
        }
    }
    response = chain.stream({"input": input_text, "context": context}, config)
    return response

context = None

def load_context():
    global context
    if context is None:
        context = Path("parsed_data/peugeot_data.txt").read_text()
load_context()

# Afficher l'historique des messages
history = st.session_state.get('chat_history', InMemoryChatMessageHistory())

if isinstance(history, InMemoryChatMessageHistory):
    for message in history.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
else:
    print("Error: chat_history is not an instance of InMemoryChatMessageHistory.")

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
        response = run_chain(user_input, context, session_id="peugeot_expert")
        st.write(response)
    
    # Ajouter la réponse de l'IA à l'historique
    add_message_to_history(AIMessage(content=response))
