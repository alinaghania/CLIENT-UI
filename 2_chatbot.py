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
st.set_page_config(page_title="BOB")
st.title("B.O.B")

# Session state pour l'historique des chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    


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
                    st.write(f"<span style='color:red;font-weight:bold'>First token received in {(first_token_time - start_time).total_seconds():.2f} seconds.</span>", unsafe_allow_html=True)
                yield token
            end_time = datetime.now()
            total_elapsed = (end_time - start_time).total_seconds()
            streaming_elapsed = (end_time - first_token_time).total_seconds() if first_token_time else 0
            st.write(f" Total response time: {total_elapsed:.2f} seconds.")
            st.write(f"Streaming time: {streaming_elapsed:.2f} seconds.")
        
        return streaming_wrapper()
    
    return wrapper_measure_time

# Fonction pour vérifier et ajouter les messages tout en respectant l'alternance
def add_message_to_history(message):
    history = st.session_state.chat_history
    if len(history.messages) == 0 or type(history.messages[-1]) != type(message):
        history.add_message(message)

# Fonction pour créer et initialiser la chaîne
def initialize_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are Bob, a Peugeot expert. Answer the following questions as best you can from this context <context>{context}</context>."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    bedrock_llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    
    chain = prompt | bedrock_llm | StrOutputParser()
    
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

# Charger le contexte du document
context = Path("parsed_data/peugeot_data.txt").read_text()

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

# Exemples de questions
st.markdown('<div class="QUESTIONS EXAMPLES">', unsafe_allow_html=True)
questions = [
    "What's the price to charge?",
    "Hello, I want to know what are the best applications?",
    "What is the customer satisfaction rate among French users who switched to electric vehicles?",
    "Can you provide a brief history of Peugeot's electric vehicles?",
    "What are the main factors influencing the autonomy of an electric vehicle?",
    "Tell me what's the price to charge my e-208, and then the time to recharge on a born."
]

for i, question in enumerate(questions, start=1):
    with st.expander(f"Question {i}"):
        st.write(question)
