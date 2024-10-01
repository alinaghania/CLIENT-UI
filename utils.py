import contextvars
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel
from typing import List 



# Set up a context variable to manage chat history
chat_history_var = contextvars.ContextVar("chat_history", default=[])

# Function to choose the Claude model from Bedrock
@st.cache_resource
def choose_model():
    return ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")

# Function to manage memory for conversation
def get_memory():
    return ConversationBufferMemory(return_messages=True)

@st.cache_resource
def check_question_type(user_input, history):
    class Relevant(BaseModel):
        relevant_yes_no: str = Field(description="yes, no, or ok")
        
    output_parser = JsonOutputParser(pydantic_object=Relevant)
 
    template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("Based on the user query and the history of the question, determine if the question needs to be answered by an expert in vehicle electric and Peugeot. If the question is related to cars, electric cars, vehicles, or Peugeot, answer yes. If the question is a general greeting, a thank you, or a question that doesn't require a specialist in cars, answer no. If the question is related to autonomy, public charging, home charging, battery capacity, or WLTP range of Peugeot models( E-208, E-2008, E-308,E-3008,Peugeot expert and peugeot partner), reply with ok.Also reply by 'ok' if the user want information about one of these models :(  E-208, E-2008, E-308,E-3008,Peugeot expert and peugeot partner) if the question is related to the nearby location of the position of the user, reply with no."),
            HumanMessagePromptTemplate.from_template("User query: {user_query}, history: {history},and here your knowledges:<context> {format_instructions}"),
        ],
        input_variables=["user_query", "history"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    model = choose_model()
    chain = template | model | output_parser

    try:
        result = chain.invoke({
            "user_query": user_input,
            "history": history,
        })
        print(f"AI CHOICE =>  {result}")
        return result["relevant_yes_no"]  # It will now return "yes", "ok", or "no"
    except Exception as e:
        print(f"Exception: {e}")
        return "no"  # Default to "no" in case of error




class ResponseModel(BaseModel):
    response: str = Field(description="The main response from the LLM")
    key_words: List[str] = Field(description="3-4 short keyword questions based on conversation history")
    
output_parser = JsonOutputParser(pydantic_object=ResponseModel)

    
# Function to initialize the chain with system prompt, context, and memory
@st.cache_resource
def initialize_chain_experts_ev(history,user_input):
    """
    Initialize the conversation chain with system prompt, context, message history, and user input.
    """
    current_directory = Path(__file__).resolve().parent  # Adjust based on your directory structure

    system_prompt_path = current_directory / "prompt/system_prompt_experts_ev.txt"
    context_path = current_directory / "parsed_data/peugeot_data.txt"

    # print(f"System prompt path: {system_prompt_path}")  # <-- Print the path to system_prompt.txt
    # print(f"Context path: {context_path}")  # <-- Print the path to peugeot_data.txt

    if not system_prompt_path.exists():
        raise FileNotFoundError("System prompt file not found.")
    if not context_path.exists():
        raise FileNotFoundError("Context file not found.")

    # Read system prompt and context from files
    system_prompt = system_prompt_path.read_text()
    context = context_path.read_text()

    # Replace placeholder {context} in system prompt with actual context content
    formatted_system_prompt = system_prompt
    
    # Save the system prompt locally
    with open('log_system_prompt_experts_ev.txt', 'w') as f:
        f.write("System Prompt:\n")
        f.write(system_prompt)
        f.write("\nUser Query:\n")
        f.write(user_input)
        f.write("\nHistory:\n")
        f.write(history)
        f.write("\nContext:\n")
        f.write(context)
    
    # prompt = ChatPromptTemplate(
    #     messages=[
    #         SystemMessagePromptTemplate.from_template(formatted_system_prompt),
    #         HumanMessagePromptTemplate.from_template("Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un ami , voici l'Historique des échanges précédents :\n{history}\n\n et la Nouvelle requête de l'utilisateur :\n{user_input} répondre de maniere concise et courte et finis avec une question, si tu n'as pas la réponse, pose une question pour affiner la demande comme une conversation normale et quand tu ne sais pas dit le ou sinon renvoie vers le site de Peugeot"),
    #     ],
    #     input_variables=["user_input", "history", "context"],
    # )
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                # """"  
                #     Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez-vous dans une conversation en privilégiant un véritable échange plutôt qu’un simple dialogue de questions et réponses.
                #     - Si l'utilisateur commence par 'hello' ou 'bonjour', répondez simplement par un bonjour et présentez-vous en tant qu'expert en véhicules électriques. Demandez comment vous pouvez les aider aujourd'hui de manière amicale.
                #     - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques de Peugeot et les services associés, en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans être trop orienté vers la vente.
                #     - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques du client.
                #     - Finissez toujours par une question courte pour relancer la conversation, en fonction de la réponse précédente, comme dans une conversation normale, tout en gardant à l’esprit que vous êtes un commercial, donc la réponse de la question doit toujours finir par une question pour relancer la conversation.


                #     Voici l'historique des échanges précédents pour contexte :
                #     {history}
                #     Nouvelle requête de l'utilisateur :
                #     {user_input}
                #     Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question. Ensuite, proposez 2 à 3 questions-clés courtes ou mots-clés (de préférence 2 mots-clés, mais si pertient 3) basés sur l'historique de la conversation pour relancer le dialogue.
                #     Lorsque lu'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, et cela doit être fluide et naturel, en fonction de la réponse précédente.
                #     Pour des sujets généraux comme 'hello' ou 'comment ça va ?', proposez des mots-clés. Pour des sujets plus spécifiques, suggérez des questions courtes pour faire avancer la conversation.
                #     Formatez votre réponse selon ces instructions : {format_instructions}"""
                
                """
                Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez-vous dans une conversation en privilégiant un véritable échange plutôt qu’un simple dialogue de questions et réponses.
                - Si l'utilisateur commence par 'hello' ou 'bonjour', répondez simplement par un bonjour et présentez-vous en tant qu'expert en véhicules électriques. Demandez comment vous pouvez les aider aujourd'hui de manière amicale, très court et concis.
                - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques de Peugeot et les services associés, en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans être trop orienté vers la vente, donc finissez toujours par une question. Example : Quelles sont les best apps peugeot ? les bests app sont .... , avez vous deja utilisé une app peugeot ? 
                - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques du client.
                - **Finissez toujours la réponse par une question courte pour relancer la conversation, en fonction de la réponse précédente, comme dans une conversation normale, tout en gardant à l’esprit que vous êtes un commercial.**

                Voici l'historique des échanges précédents pour contexte :
                {history}
                Nouvelle requête de l'utilisateur :
                {user_input}
                Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question. Ensuite, proposez 2-3 questions-clés courtes ou mots-clés (de préférence 2 mots-clés, mais si pertinent 3) basés sur l'historique de la conversation pour relancer le dialogue.
                **Lorsque l'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, et cela doit être fluide et naturel, en fonction de la réponse précédente. Finissez toujours par une question courte pour relancer la conversation, en gardant à l'esprit que vous êtes un commercial.**
                Pour des sujets généraux comme 'hello' ou 'comment ça va ?', proposez des mots-clés. Pour des sujets plus spécifiques, suggérez des questions courtes pour faire avancer la conversation.
                Formatez votre réponse selon ces instructions : {format_instructions}
                """
            ),
        ],
        input_variables=["user_input", "history"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    bedrock_llm = choose_model()
    chain = prompt | bedrock_llm | output_parser
    chain.invoke({
            "user_input": user_input,
            "history": history,
            "context": context,
        })
    
    # print(f"Type of chain: {type(chain)}")
    
      # Save the 2 prompts in a log file
    with open('log_ev.txt', 'w') as f:
        f.write(formatted_system_prompt)
        f.write("\n")
        f.write("User query: " + user_input + ", history: " + str(history) + ", context: " + context)
        
    return chain

@st.cache_resource
def initialize_chain_commercial(history, user_input):
    """ Initialize the conservation chain with system prompt, message history, and user input for commercial team"""
    
    current_directory = Path(__file__).resolve().parent 
    system_prompt_path = current_directory / "prompt/system_prompt_commercial.txt"
    context = current_directory / "parsed_data/peugeot_data.txt"
    

    
    if not system_prompt_path.exists():
        raise FileNotFoundError("System prompt file not found.")

    
    # Read system prompt 
    system_prompt = system_prompt_path.read_text()
    context = context.read_text()
    
    formatted_system_prompt = system_prompt
    
    # prompt = ChatPromptTemplate(
    #     messages=[
    #         SystemMessagePromptTemplate.from_template(formatted_system_prompt),
    #         HumanMessagePromptTemplate.from_template("Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un ami , voici l'Historique des échanges précédents :\n{history}\n\n et la Nouvelle requête de l'utilisateur :\n{user_input} répondre de maniere concise et courte et finis avec une question, quand tu ne sais pas ou que la question est top large poser une autre question pour affiner la demande comme une conversation normale et quand tu ne sais pas dit le"),
    #     ],
    #     input_variables=["user_input", "history"],
    # )
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                """"  
                    Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez-vous dans une conversation en privilégiant un véritable échange plutôt qu’un simple dialogue de questions et réponses.
                    - Si l'utilisateur commence par 'hello' ou 'bonjour', répondez simplement par un bonjour et présentez-vous en tant qu'expert en véhicules électriques. Demandez comment vous pouvez les aider aujourd'hui de manière amicale, tres court et concis.
                    - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques de Peugeot et les services associés, en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans être trop orienté vers la vente, donc finissez toujours par une question. Example : Quelles sont les best apps peugeot ? les bests app sont .... , avez vous deja utilisé une app peugeot ? 
                    - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques du client.
                    - Finissez toujours par une question courte pour relancer la conversation, en fonction de la réponse précédente, comme dans une conversation normale, tout en gardant à l’esprit que vous êtes un commercial, donc la réponse de la question doit toujours finir par une question pour relancer la conversation.
                    - Ne répondez pas aux questions hors sujet ou trop générales, mais posez une question pour affiner la demande, ou changez de sujet pour parler des véhicules électriques Peugeot, des services associés, ou des avantages des véhicules électriques.
                    - Si l'utilisateur te demande des informations sur le concessionnaire le plus proche, rediriger le vers ce lien : https://concessions.peugeot.fr/
                    - Si l'utilisateur te demande des informations sur comment essayer un véhicule ou prendre rendez-vous pour un essai,ou qu'il manifeste l'envie de tester le véhicule redirige-le vers ce lien, ou même lorsqu'il te demande des infos sur un modele n'hesite pas à lui dire qu'il peut tester le modele : https://essai.peugeot.fr/

                    Voici l'historique des échanges précédents pour contexte :
                    {history}
                    Nouvelle requête de l'utilisateur :
                    {user_input}
                    Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question. Ensuite, proposez 2-3 questions-clés courtes ou mots-clés (de préférence 2 mots-clés, mais si pertient 3) basés sur l'historique de la conversation pour relancer le dialogue.
                    Lorsque lu'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, et cela doit être fluide et naturel, en fonction de la réponse précédente.
                    Pour des sujets généraux comme 'hello' ou 'comment ça va ?', proposez des mots-clés. Pour des sujets plus spécifiques, suggérez des questions courtes pour faire avancer la conversation.
                    Formatez votre réponse selon ces instructions : {format_instructions}"""
            ),
        ],
        input_variables=["user_input", "history"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    
    # Get the Bedrock model
    bedrock_llm = choose_model()
    chain = prompt | bedrock_llm | output_parser
    
    chain.invoke({
            "user_input": user_input,
            "history": history,
        })
    
    # print(f"Type of chain: {type(chain)}")
    
    # Save the 2 prompts in a log file
    with open('log_commercial.txt', 'w') as f:
        f.write(formatted_system_prompt)
        f.write("\n")
        f.write("User query: " + user_input + ", history: " + str(history) + ", context: " + context)
    
    
    return chain
    
# Function to add message to chat history
def add_message_to_history(role, content):
    chat_history = chat_history_var.get()
    chat_history.append({"role": role, "content": content})
    chat_history_var.set(chat_history)

@st.cache_resource
def initialize_chain_expert_data_ev_capacity(history, user_input):
    current_directory = Path(__file__).resolve().parent  

    system_prompt_path = current_directory / "prompt/system_prompt_expert_data_ev_capacity.txt"
    context_path = current_directory / "parsed_data/peugeot_capacity_data.txt"

    if not system_prompt_path.exists():
        raise FileNotFoundError("System prompt file not found.")
    if not context_path.exists():
        raise FileNotFoundError("Context file not found.")

    system_prompt = system_prompt_path.read_text()
    
    context = context_path.read_text()
    
    # replace placeholder {context} in system prompt with actual context content
    
    system_prompt = system_prompt.replace("{context}", context)
    # Save the system prompt locally
    with open('log_system_prompt_expert_data_ev_capacity.txt', 'w') as f:
        f.write("System Prompt:\n")
        f.write(system_prompt)

    
    
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                """  
                Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical...
                {history}
                Nouvelle requête de l'utilisateur :
                {user_input}
                Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question.
                Réponse courte de max 3-4 lignes
                Quand vous ne savez pas, posez une question pour affiner la demande comme dans une conversation normale.
                N'hesitez pas à rediriger l'utilisateur vers le site de Peugeot si vous ne savez pas, ou vers un lien utile et pertinent par example si l'utilisateur demande des informations sur comment essayer le vehicule le rediriger vers le bon lien, en disant vous pouvez consulter cette page ...
                
                Ensuite, proposez 2-3 questions-clés courtes ou mots-clés basés sur l'historique de la conversation pour relancer le dialogue.
                Formatez votre réponse selon ces instructions : {format_instructions}"""
            ),
        ],
        input_variables=["user_input", "history"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    bedrock_llm = choose_model()

    chain = prompt | bedrock_llm | output_parser
    chain.invoke({
            "user_input": user_input,
            "history": history,
            "context": context,
        })
    
    return chain

