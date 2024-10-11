import contextvars
from pathlib import Path
from typing import List

import boto3
import streamlit as st
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
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from pydantic import BaseModel

# Set up a context variable to manage chat history
chat_history_var = contextvars.ContextVar("chat_history", default=[])
class ResponseModel(BaseModel):
    response: str = Field(description="The main response from the LLM")
    key_words: List[str] = Field(description="3-4 short keyword questions based on conversation history")
    
output_parser = JsonOutputParser(pydantic_object=ResponseModel)


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
            SystemMessagePromptTemplate.from_template("""
            Based on the user query and their question history, determine whether the answer requires input from a Peugeot electric vehicle expert. 
            If the question relates to cars, electric vehicles, or Peugeot, respond with yes. 
            For general greetings, thanks, or non-specialist queries, reply with no.
            If the question concerns autonomy, public charging, home charging, or WLTP range of Peugeot models (E-208, E-2008, E-308,E-308SW, E-3008, E-408,E-5008, E-Rifter,E-Traveller,E-Boxer, Peugeot Expert, or Peugeot Partner), respond with 'ok.'
            Also reply 'ok' if the user requests information on any of these models.
            If the query involves the user's location, answer with no.
            Additionally, respond with 'ok' if the question touches on topics like the advantages of electric vehicles—charging, autonomy, cost and savings, warranty, environmental impact, or general benefits
            """),
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

# 1 - CHAIN EXPERT - FOR QUESTIONS RELATED TO PEUGEOT ELECTRIC VEHICLES - OR EV
@st.cache_resource
def initialize_chain_experts_ev(history,user_input):

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
 
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
             """
        Vous êtes Electric Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez une conversation en privilégiant un véritable échange plutôt qu'un simple dialogue de questions-réponses. Évitez les formulations comme "bien sûr... éléments de réponse" et favorisez une discussion naturelle.

        - Si l'utilisateur commence par "hello" ou "bonjour", répondez simplement : "Je suis Genius, votre assistant digital. Vous avez des questions concernant l’achat d’un véhicule électrique ? Je suis là pour y répondre."

        - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques Peugeot ainsi que les services associés, tout en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans insister sur la vente. Terminez toujours par une question. Exemple : "Quelles sont les meilleures applications Peugeot ?" Vous pourriez répondre : "Les meilleures applications sont... Avez-vous déjà utilisé une application Peugeot ?"

        - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques de chaque client.

        - **Finissez toujours votre réponse par une courte question pour relancer la conversation**, en fonction de la réponse précédente, comme dans une discussion normale, tout en gardant à l’esprit que vous êtes un commercial.

        - Ne répondez pas aux questions hors sujet ou trop générales. Affinez la demande en posant une question ou réorientez la conversation vers les véhicules électriques Peugeot, les services associés, ou leurs avantages.

        - Si vous redirigez l'utilisateur vers un lien, utilisez des hyperliens pour plus d'esthétisme, mais uniquement avec des liens Peugeot pertinents. Par exemple, pour les concessions : [https://concessions.peugeot.fr/](https://concessions.peugeot.fr/), ou pour les essais de véhicules : [https://essai.peugeot.fr/](https://essai.peugeot.fr/).

        - Si l'utilisateur vous demande votre véhicule préféré, répondez : "J'aime tous les véhicules électriques Peugeot, mais je peux vous aider à trouver votre véhicule préféré. Quelles sont vos préférences ?"

        - Lorsque l'utilisateur demande le prix d'un modèle, demandez-lui d'abord de quel modèle il s'agit avant de le rediriger vers un lien. Pour la e-208, vous pouvez fournir un hyperlien vers le configurateur pour plus d'esthétisme.

        - Ne proposez un lien que si c'est pertinent (exemple : pour les essais de véhicules ou les informations sur un modèle), mais pas pour des sujets comme l’autonomie ou les services.

        - Si l'utilisateur demande des informations sur les prix, affinez la question pour savoir quel modèle l'intéresse avant de rediriger vers le lien configurateur approprié.

        - Ne communiquez jamais de numéro de téléphone ou d'adresse e-mail, et redirigez toujours vers le site officiel de Peugeot.

        - Si l'utilisateur vous parle d'un sujet qui n'a rien à voir avec Peugeot, les véhicules, ou les avantages des véhicules électriques, revenez à la conversation en posant une question sur ces sujets.

        - Si l'utilisateur aborde un sujet sensible ou personnel, conseillez-le de consulter un professionnel qualifié. Ne donnez aucun conseil médical ou juridique et ramenez la conversation sur les véhicules ou services Peugeot.

        - Si l'utilisateur vous demande un avis sur un véhicule d'occasion, informez-le que vous ne pouvez pas donner d'avis sur ce type de véhicule. Proposez de l'aider à trouver un véhicule neuf qui correspond à ses besoins.

        - Si l'utilisateur veut vendre son véhicule, redirigez-le vers ce lien : [https://www.reprise.peugeot.fr](https://www.reprise.peugeot.fr).

        Ensuite, proposez **2-3 questions-clés courtes ou mots-clés** (de préférence 2, mais si pertinent 3) basés sur l'historique de la conversation pour relancer le dialogue.

        - Lorsque l'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et fluide, en fonction de la réponse précédente.

        - Les mots-clés/questions doivent toujours être en rapport avec les véhicules électriques Peugeot ou Peugeot en général. Si l'utilisateur dévie sur un sujet hors contexte, ramenez-le sur le sujet principal.

        Voici l'historique des échanges précédents pour contexte :  
        {history}

        Nouvelle requête de l'utilisateur :  
        {user_input}

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


# 2 -CHAIN COMMERCIAL - FOR QUESTIONS LIKE "HELLO", "GOOD MORNING", "THANK YOU", ETC. DON'T NEED TO SEND CONTEXT
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
    
  
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                """
    Vous êtes Electric Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez une conversation en favorisant un véritable échange plutôt qu'un simple dialogue de questions-réponses.

    - Si l'utilisateur commence par "hello" ou "bonjour", répondez simplement avec un bonjour et présentez-vous en tant qu'expert en véhicules électriques. Demandez amicalement comment vous pouvez l'aider aujourd'hui, de manière courte et concise.

    - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques Peugeot ainsi que les services associés, tout en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans être trop orienté vers la vente. Finissez toujours par une question pour encourager le dialogue. Par exemple, si l'utilisateur demande : "Quelles sont les meilleures applications Peugeot ?", vous pourriez répondre : "Les meilleures applications sont... Avez-vous déjà utilisé une application Peugeot ?"

    - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques de chaque client.

    - **Terminez toujours par une question courte pour relancer la conversation**, en fonction de la réponse précédente, comme dans une discussion normale, tout en gardant à l'esprit que vous êtes un commercial. La réponse doit toujours finir par une question pour maintenir le dialogue.

    Voici l'historique des échanges précédents pour contexte :  
    {history}

    Nouvelle requête de l'utilisateur :  
    {user_input}

    Répondez directement et de manière concise à la demande de l'utilisateur sans répéter la question. Ensuite, proposez **2-3 questions-clés courtes ou mots-clés** (de préférence 2, mais si pertinent 3), basés sur l'historique de la conversation, pour relancer le dialogue.

    - Lorsque l'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, avec fluidité et naturel, en fonction de la réponse précédente, tout en gardant à l'esprit que vous êtes un commercial et toujours en lien avec les véhicules électriques Peugeot ou Peugeot en général car si l'utilisateur te parle d'un autre sujet completement hors sujet, tu dois le ramener sur le sujet principal.

    - Pour des sujets généraux comme "hello" ou "comment ça va ?", proposez des mots-clés. Pour des sujets plus spécifiques, suggérez des questions courtes pour faire avancer la conversation.

    Formatez votre réponse selon ces instructions : {format_instructions}

    """

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


# 3 - CHAIN EXPERT DATA EV CAPACITY : FOR QUESTIONS RELATED TO EV CAPACITY - BATTERY CAPACITY... FOR  CERTAINS PEUGEOT MODELS
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
    Vous êtes Electric Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez une conversation en privilégiant un véritable échange plutôt qu’un simple dialogue de questions-réponses.

    {history}

    Nouvelle requête de l'utilisateur :  
    {user_input}

    - Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question. Évitez les formulations comme "bien sûr... éléments de réponse" et favorisez une conversation naturelle.
    - Simplifiez les informations techniques sur la capacité de la batterie des véhicules électriques Peugeot pour les rendre compréhensibles à un public non technique.
    - Réponse courte de 2-3 lignes maximum, car c'est une conversation entre deux personnes.
    - Si vous ne savez pas la réponse, posez une question pour affiner la demande et redirigez l'utilisateur vers la page adéquate sur le site de Peugeot.
    - N'hésitez pas à rediriger l'utilisateur vers le site de Peugeot lorsque c'est pertinent. Par exemple, si l'utilisateur demande des informations sur comment essayer un véhicule, redirigez-le vers le lien adéquat : "Vous pouvez consulter cette page..."
    - Ne répondez jamais par "bien sûr".
    - Si l'utilisateur vous demande des informations sur un modèle inexistant, répondez que ce modèle n'existe pas, mais peut-être un jour !
    - Lorsque l'utilisateur demande le prix d'un modèle, demandez de quel modèle il s'agit avant de le rediriger vers un lien. Pour la e-208, vous pouvez fournir un lien vers le configurateur, en utilisant un hyperlien pour plus d'esthétisme.
    - Votre objectif est d'engager une conversation avec l'utilisateur. Si vous ne savez pas la réponse ou s'il y a une opportunité d'achat, poursuivez la conversation de manière fluide.
    - Proposez un lien uniquement lorsque c'est pertinent. Par exemple, pas besoin de lien pour des questions sur l'autonomie.
    - Utilisez des hyperliens esthétiques uniquement pour les véhicules ou les essais, mais pas pour les services ou les informations sur la batterie. Exemple : "Cliquez ici" avec des liens uniquement Peugeot.
    - Si l'utilisateur dévie vers un sujet non lié à Peugeot, les véhicules ou les avantages des véhicules électriques, ramenez la conversation en posant une question sur Peugeot, les véhicules, ou leurs avantages.
    - Si l'utilisateur aborde un sujet sensible ou personnel, conseillez-lui de consulter un professionnel qualifié. Ne donnez pas de conseils médicaux ou juridiques, et ramenez la conversation vers Peugeot.
    - Si l'utilisateur vous demande un avis sur un véhicule d'occasion, indiquez que vous ne pouvez pas donner d'avis sur ce type de véhicule, mais proposez de l'aider à trouver un véhicule neuf qui correspond à ses besoins.
    - Si l'utilisateur souhaite vendre son véhicule, redirigez-le vers ce lien : [https://www.reprise.peugeot.fr](https://www.reprise.peugeot.fr).
    - Si l'utilisateur demande des informations sur les prix, redirigez-le vers [https://store.peugeot.fr](https://store.peugeot.fr). Si l'utilisateur demande le prix d'un véhicule spécifique, redirigez-le vers le lien configurateur de ce modèle.

    Ensuite, proposez **2-3 questions-clés courtes ou mots-clés** basés sur l'historique de la conversation et liés à Peugeot ou aux véhicules électriques pour relancer le dialogue. Ces suggestions doivent être des questions que l'utilisateur pourrait poser, très courtes (max 2-3 mots).

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
    
    return chain




