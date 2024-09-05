import functools
from datetime import datetime
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path

# Fonction pour mesurer le temps d'exécution et les tokens pendant le streaming
def measure_metrics(func):
    @functools.wraps(func)
    def wrapper_measure_metrics(*args, **kwargs):
        start_time = datetime.now()
        first_token_time = None

        def streaming_wrapper():
            nonlocal first_token_time
            for token in func(*args, **kwargs):
                if first_token_time is None:
                    first_token_time = datetime.now()
                yield token

        result = "".join(streaming_wrapper())

        end_time = datetime.now()
        total_elapsed = (end_time - start_time).total_seconds()
        first_token_time_elapsed = (first_token_time - start_time).total_seconds() if first_token_time else None
        streaming_elapsed = (end_time - first_token_time).total_seconds() if first_token_time else None

        time_metrics = {
            "First Token Time (s)": first_token_time_elapsed,
            "Total Elapsed Time (s)": total_elapsed,
            "Streaming Elapsed Time (s)": streaming_elapsed
        }

        return result, time_metrics

    return wrapper_measure_metrics

# Fonction pour vérifier et ajouter les messages tout en respectant l'alternance
def add_message_to_history(message):
    history = st.session_state.chat_history
    if len(history.messages) == 0 or type(history.messages[-1]) != type(message):
        history.add_message(message)

# Fonction pour créer et initialiser la chaîne
# def initialize_chain():
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are Bob, a Peugeot expert. Answer the following questions as best you can from this context <context>{context}</context>."),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#     ])

#     bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")

#     chain = prompt | bedrock_llm | StrOutputParser()

#     wrapped_chain = RunnableWithMessageHistory(
#         chain,
#         lambda: st.session_state.chat_history,
#         history_messages_key="chat_history",
#     )

#     return wrapped_chain, bedrock_llm


# Fonction pour choisir le modèle sur Bedrock
def choose_model():
    # Choix du modèle Claude 3.5 Sonnet depuis Amazon Bedrock
    bedrock_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    return bedrock_llm



# Fonction pour initialiser la chaîne en lisant le prompt depuis un fichier
def initialize_chain():
    # Lire le prompt système depuis le fichier externe "prompt/system_prompt.txt"
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

    return wrapped_chain, bedrock_llm


# Appliquer le décorateur pour mesurer les métriques
@measure_metrics
def run_chain(input_text, context):
    chain, bedrock_llm = initialize_chain()

    full_input = f"{context}\n{input_text}"
    input_tokens = bedrock_llm.get_num_tokens(full_input)

    response_stream = chain.stream({"input": input_text, "context": context})
    return "".join([chunk for chunk in response_stream])

# Fonction principale pour gérer l'interaction et sauvegarder les métriques
def process_input(input_text, context):
    response, time_metrics = run_chain(input_text, context)
    
    chain, bedrock_llm = initialize_chain()
    output_tokens = bedrock_llm.get_num_tokens(response)
    input_tokens = bedrock_llm.get_num_tokens(f"{context}\n{input_text}")
    
    input_cost = (input_tokens / 1_000_000) * 3
    output_cost = (output_tokens / 1_000_000) * 15
    total_cost = input_cost + output_cost
    
    metrics = {
        "User Question": input_text,
        "AI Response": response,
        "Input Tokens": input_tokens,
        "Output Tokens": output_tokens,
        "Input Cost ($)": input_cost,
        "Output Cost ($)": output_cost,
        "Total Cost ($)": total_cost,
        **time_metrics
    }
    
    st.session_state.metrics.append(metrics)
    
    return response

# Fonction pour sauvegarder les résultats sous forme de CSV
def save_results_to_csv():
    df = pd.DataFrame(st.session_state.metrics)
    return df.to_csv(index=False).encode("utf-8")
