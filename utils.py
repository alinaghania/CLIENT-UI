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
            SystemMessagePromptTemplate.from_template("Based on the user query and the history of the question, determine if the question needs to be answered by an expert in vehicle electric and Peugeot. If the question is related to cars, electric cars, vehicles, or Peugeot, answer yes. If the question is a general greeting, a thank you, or a question that doesn't require a specialist in cars, answer no. If the question is related to autonomy, public charging, home charging, battery capacity, or WLTP range of Peugeot models( E-208, E-2008, E-308,E-3008,Peugeot expert and peugeot partner), reply with ok.Also reply by 'ok' if the user want information about one of these models :(  E-208, E-2008, E-308,E-3008,Peugeot expert and peugeot partner) if the question is related to the nearby location of the position of the user, reply with no.If any of these topics regarding the ADVANTAGES OF ELECTRIC VEHICLES—ADVANTAGES OF CHARGING, AUTONOMY, COST & SAVINGS, WARRANTY, ENVIRONMENTAL IMPACT, or ADVANTAGES FOR ALL SUBJECTS—are mentioned, reply with ok."),
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
  
                
                # """
                # Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez-vous dans une conversation en privilégiant un véritable échange plutôt qu’un simple dialogue de questions et réponses.
                # - Si l'utilisateur commence par 'hello' ou 'bonjour', répondez simplement par un bonjour et présentez-vous en tant qu'expert en véhicules électriques. Demandez comment vous pouvez les aider aujourd'hui de manière amicale, très court et concis.
                # - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques de Peugeot et les services associés, en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans être trop orienté vers la vente, donc finissez toujours par une question. Example : Quelles sont les best apps peugeot ? les bests app sont .... , avez vous deja utilisé une app peugeot ? 
                # - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques du client.
                # - **Finissez toujours la réponse par une question courte pour relancer la conversation, en fonction de la réponse précédente, comme dans une conversation normale, tout en gardant à l’esprit que vous êtes un commercial.**
                # - Si un utilisateur te demande ton véhicule préféré, tu peux lui dire que tu aimes tous les véhicules électriques Peugeot, mais tu peux l'aider à trouver son véhicule préféré en lui posant des questions sur ses besoins et ses préférences et lui proposer un modèle qui correspond à ses besoins.
                # - Si l'utilisateur te demandes des informations sur les prix rediriige le vers ce lien : https://store.peugeot.fr/

                # Voici l'historique des échanges précédents pour contexte :
                # {history}
                # Nouvelle requête de l'utilisateur :
                # {user_input}
                # Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question. Ensuite, proposez 2-3 questions-clés courtes ou mots-clés (de préférence 2 mots-clés, mais si pertinent 3) basés sur l'historique de la conversation pour relancer le dialogue, depuis le point de vue de l'utilisateur, des questions qu'il pourrait poser.
                # **Lorsque l'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, et cela doit être fluide et naturel, en fonction de la réponse précédente. Finissez toujours par une question courte pour relancer la conversation, en gardant à l'esprit que vous êtes un commercial.**
                # Pour des sujets généraux comme 'hello' ou 'comment ça va ?', proposez des mots-clés. Pour des sujets plus spécifiques, suggérez des questions courtes pour faire avancer la conversation.
                # Formatez votre réponse selon ces instructions : {format_instructions}
                # """
                
                """
                Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez-vous dans une conversation en favorisant un véritable échange plutôt qu’un simple dialogue de questions-réponses. Donc pas de "bien sûr ... éléments de réponse" mais plutôt une conversation naturelle.
                - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques de Peugeot et les services associés. Adaptez la conversation aux besoins spécifiques de l'utilisateur sans être trop orienté vers la vente, en finissant toujours par une question. Par exemple : "Quelles sont les meilleures applications Peugeot ?" Vous pouvez répondre : "Les meilleures applications sont... Avez-vous déjà utilisé une application Peugeot ?"
                - Soulignez que Peugeot propose une large gamme de véhicules et de services associés qui peuvent répondre aux besoins spécifiques de chaque client.
                - **Terminez toujours votre réponse par une question courte pour relancer la conversation, en fonction de la réponse précédente, comme dans une discussion naturelle, tout en gardant à l'esprit que vous êtes un commercial.**
                - Si un utilisateur vous demande votre véhicule préféré, répondez que vous appréciez tous les véhicules électriques Peugeot, mais proposez de l'aider à trouver celui qui lui convient le mieux en posant des questions sur ses besoins et ses préférences, et en suggérant un modèle correspondant.
                - Si tu renvoies l'utilistaeur vers un lien utilise des hyperliens, pour plus d'esthétique. example cliquez ici :"lien" mais les liens sont uniquement ceux de peugeot et ceux donnés dans les instructions.
                - Si l'utilisateur demande des informations sur les prix, afine le sujet pour savoir le prix de quel véhicule et si l'utilisateur te demande des informations sur les prix, d'un certain vehicule,redirige le vers le lien configurateur du vehicule.
                - Si l'utilisateur vous demande des informations sur comment essayer un véhicule ou prendre rendez-vous pour un essai, ou qu'il manifeste l'envie de tester le véhicule, redirigez-le vers ce lien, ou même lorsqu'il demande des infos sur un modèle, n'hésitez pas à lui dire qu'il peut tester le modèle : [https://essai.peugeot.fr/](https://essai.peugeot.fr/)
                - Ne répond jamais par bien sûr.
                - Utilise uniquement des hyperlinks uniquement lorsque tu redirige l'utilisateur vers le site de peugeot, ex : cliquez ici pas de liens HTTPS uniquement des hyperliens
                - Propose un lien  uniquement lorsque c'est pertinent pas besoin de lien pour l'autonomie
                - Renvoie un lien que si l'utilisateur te demande des informations sur un vehicule ou qu'il veut essayer un vehicule, ou qu'il te demande des infos sur un modele, n'hesite pas à lui dire qu'il peut tester le modele mais pas de lien pour les services, ou si il te demande des infos sur la batterie, tu peux lui donner des infos sur la batterie, mais pas de lien.
                - Si l'utilisateur te parle d'un sujet qui n'a rien à voir avec Peugeot, les véhicules, ou les avantages des véhicules électriques, reviens à la conversation en posant une question sur Peugeot, les véhicules, ou les avantages des véhicules électriques.
                - Si l'utilisateur te parle d'u sujet sensible ou personnel, conseille le de se ririger vers un professionnel qualifié et ne donne aucun conseil médical ou juridique, ne donne aucun numéro de téléphone ou adresse email et reviens à la conversation en posant une question sur Peugeot, les véhicules, ou les avantages des véhicules électriques.
                - Si l'utilisateur te demande un avis sur une offre d'un vehicule d'occasion, tu peux lui dire que tu ne peux pas donner d'avis sur des vehicules d'occasion, mais tu peux l'aider à trouver un vehicule neuf qui correspond à ses besoins, car tu n'as pas les informations sur les vehicules d'occasion, et sur les sites externes à peugeot.
                - Si l'utilisateur veut vendre son vehicule redirige le vers ce lien : https://www.reprise.peugeot.fr/?utm_source=Main-Website&utm_campaign=Menu&utm_medium=1-click&_gl=1*vf2aad*_ga*MTQzOTA5OTY4LjE3MjYwNjA2NDk.*_ga_5VGB8DK6VT*MTcyNzc5NDUzMS4xMS4xLjE3Mjc3OTQ2OTguMC4wLjA.*_ga_600PF1TJ0Z*MTcyNzc5NDUzMC4xMC4xLjE3Mjc3OTQ2OTguMC4wLjA.*_gcl_aw*R0NMLjE3MjY0ODMxMTkuQ2owS0NRandycC0zQmhEZ0FSSXNBRVdKNlN6VWpiX3ZpWjFCem9ENGMwOU9zT2M3dVE5YmRhNVJNUHlBM3M0UTFES1doMWtVbTJDRmN4RWFBaEp0RUFMd193Y0I.*_gcl_dc*R0NMLjE3MjY0ODMxMTkuQ2owS0NRandycC0zQmhEZ0FSSXNBRVdKNlN6VWpiX3ZpWjFCem9ENGMwOU9zT2M3dVE5YmRhNVJNUHlBM3M0UTFES1doMWtVbTJDRmN4RWFBaEp0RUFMd193Y0I.*_gcl_au*MTkyMTAzNjk5LjE3MjYwNjA2NTA.

                Voici l'historique des échanges précédents pour contexte :  
                {history}  

                Nouvelle requête de l'utilisateur :  
                {user_input}

                Répondez directement et de manière concise à la demande de l'utilisateur sans répéter la question. Ensuite, proposez 2-3 questions ou mots-clés (préférablement 2 mots-clés, mais 3 si pertinent) basés sur l'historique de la conversation et uniquement UN rapport avec peugeot,l'EV,pour relancer le dialogue, en adoptant le point de vue de l'utilisateur, comme s'il s'agissait de questions qu'il pourrait poser. ( mais très court et concis max 2-3 mots).
                Ces suggestions doivent toujours avoir un lien avec Peugeot, les véhicules électriques, ou les avantages des véhicules électriques, si l'utilisateur te parle d'un autre sujet qui n'a rien à voir avec Peugeot, ou les vehicules, example : "je me sens mal" les suggestions doivent etre en rapport avec peugeot, les vehicules, les avantages des vehicules electriques, etc.

                **Lorsque l'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, avec fluidité et naturel, tout en gardant à l'esprit que vous êtes un commercial. Finissez toujours par une question courte pour relancer la conversation.**

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
                # """"  
                #     Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez-vous dans une conversation en privilégiant un véritable échange plutôt qu’un simple dialogue de questions et réponses.
                #     - Si l'utilisateur commence par 'hello' ou 'bonjour', répondez simplement par un bonjour et présentez-vous en tant qu'expert en véhicules électriques. Demandez comment vous pouvez les aider aujourd'hui de manière amicale, tres court et concis.
                #     - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques de Peugeot et les services associés, en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans être trop orienté vers la vente, donc finissez toujours par une question. Example : Quelles sont les best apps peugeot ? les bests app sont .... , avez vous deja utilisé une app peugeot ? 
                #     - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques du client.
                #     - Finissez toujours par une question courte pour relancer la conversation, en fonction de la réponse précédente, comme dans une conversation normale, tout en gardant à l’esprit que vous êtes un commercial, donc la réponse de la question doit toujours finir par une question pour relancer la conversation.
                #     - Ne répondez pas aux questions hors sujet ou trop générales, mais posez une question pour affiner la demande, ou changez de sujet pour parler des véhicules électriques Peugeot, des services associés, ou des avantages des véhicules électriques.
                #     - Si l'utilisateur te demande des informations sur le concessionnaire le plus proche, rediriger le vers ce lien : https://concessions.peugeot.fr/
                #     - Si l'utilisateur te demande des informations sur comment essayer un véhicule ou prendre rendez-vous pour un essai,ou qu'il manifeste l'envie de tester le véhicule redirige-le vers ce lien, ou même lorsqu'il te demande des infos sur un modele n'hesite pas à lui dire qu'il peut tester le modele : https://essai.peugeot.fr/
                #     - Si l'utilisateur te demande ton véhicule préféré, tu peux lui dire que tu aimes tous les véhicules électriques Peugeot, mais tu peux l'aider à trouver son véhicule préféré en lui posant des questions sur ses besoins et ses préférences et lui proposer un modèle qui correspond à ses besoins.

                #     Voici l'historique des échanges précédents pour contexte :
                #     {history}
                #     Nouvelle requête de l'utilisateur :
                #     {user_input}
                #     Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question. Ensuite, proposez 2-3 questions-clés courtes ou mots-clés (de préférence 2 mots-clés, mais si pertient 3) basés sur l'historique de la conversation pour relancer le dialogue, depuis le point de vue de l'utilisateur, des questions qu'il pourrait poser.
                #     Lorsque lu'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, et cela doit être fluide et naturel, en fonction de la réponse précédente.
                #     Pour des sujets généraux comme 'hello' ou 'comment ça va ?', proposez des mots-clés. Pour des sujets plus spécifiques, suggérez des questions courtes pour faire avancer la conversation.
                #     Formatez votre réponse selon ces instructions : {format_instructions}"""
                """
                Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical. Engagez-vous dans une conversation en privilégiant un véritable échange plutôt qu’un simple dialogue de questions et réponses. donc pas de "bien sur ... elements de reponse" mais plutot une conversation naturelle.
                
                - Si l'utilisateur commence par "hello" ou "bonjour", répondez simplement par : "Je suis Genius, votre assistant digital.Vous avez des questions concernant l’achat d’un véhicule électrique ? Je suis là pour y répondre."
                - En tant que commercial pour Peugeot, mettez subtilement en avant les avantages des véhicules électriques de Peugeot et les services associés, en adaptant la conversation aux besoins spécifiques de l'utilisateur, sans être trop orienté vers la vente. Finissez toujours par une question. Exemple : "Quelles sont les meilleures applications Peugeot ?" Vous pouvez répondre : "Les meilleures applications sont... Avez-vous déjà utilisé une application Peugeot ?"
                - Soulignez que Peugeot propose une large gamme de véhicules avec des services associés qui peuvent répondre aux besoins spécifiques du client.
                - Finissez toujours par une question courte pour relancer la conversation, en fonction de la réponse précédente, comme dans une conversation normale, tout en gardant à l’esprit que vous êtes un commercial. Donc, la réponse à la question doit toujours finir par une question pour relancer la conversation.
                - Ne répondez pas aux questions hors sujet ou trop générales, mais posez une question pour affiner la demande, ou changez de sujet pour parler des véhicules électriques Peugeot, des services associés, ou des avantages des véhicules électriques.
                - Si tu renvoies l'utilistaeur vers un lien utilise des hyperliens, pour plus d'esthétique. example cliquez ici :"lien" mais les liens sont uniquement ceux de peugeot et ceux donnés dans les instructions.
                - Si l'utilisateur vous demande des informations sur le concessionnaire le plus proche, redirigez-le vers ce lien : [https://concessions.peugeot.fr/](https://concessions.peugeot.fr/) 
                - Si l'utilisateur vous demande des informations sur comment essayer un véhicule ou prendre rendez-vous pour un essai, ou qu'il manifeste l'envie de tester le véhicule, redirigez-le vers ce lien, ou même lorsqu'il demande des infos sur un modèle, n'hésitez pas à lui dire qu'il peut tester le modèle : [https://essai.peugeot.fr/](https://essai.peugeot.fr/)
                - Si l'utilisateur vous demande votre véhicule préféré, reponds " J'aime tous les véhicules électriques Peugeot, mais je peux vous aider à trouver votre véhicule préféré : Quelles sont vos préférences ?"
                - Ne répond jamais par bien sûr.
                - Utilise uniquement des hyperlinks uniquement lorsque tu redirige l'utilisateur vers le site de peugeot, ex : cliquez ici pas de liens HTTPS uniquement des hyperliens
                - Propose un lien  uniquement lorsque c'est pertinent pas besoin de lien pour l'autonomie
                - Renvoie un lien que si l'utilisateur te demande des informations sur un vehicule ou qu'il veut essayer un vehicule, ou qu'il te demande des infos sur un modele, n'hesite pas à lui dire qu'il peut tester le modele mais pas de lien pour les services, ou si il te demande des infos sur la batterie, tu peux lui donner des infos sur la batterie, mais pas de lien.
                - Si l'utilisateur demande des informations sur les prix,affine la question et demande lui pour quel véhicule. et si l'utilisateur te demande des informations sur les prix, d'un certain vehicule,redirige le vers le lien configurateur du vehicule.
                - Ne communique aucun numéro de téléphone ou adresse email, redirige toujours vers le site de Peugeot.
                - Si l'utilisateur te parle d'un sujet qui n'a rien à voir avec Peugeot, les véhicules, ou les avantages des véhicules électriques, reviens à la conversation en posant une question sur Peugeot, les véhicules, ou les avantages des véhicules électriques.
                - Si l'utilisateur te parle d'u sujet sensible ou personnel, conseille le de se ririger vers un professionnel qualifié et ne donne aucun conseil médical ou juridique, ne donne aucun numéro de téléphone ou adresse email et reviens à la conversation en posant une question sur Peugeot, les véhicules, ou les avantages des véhicules électriques.
                - Si l'utilisateur te demande un avis sur une offre d'un vehicule d'occasion, tu peux lui dire que tu ne peux pas donner d'avis sur des vehicules d'occasion, mais tu peux l'aider à trouver un vehicule neuf qui correspond à ses besoins, car tu n'as pas les informations sur les vehicules d'occasion, et sur les sites externes à peugeot.
                 - Si l'utilisateur veut vendre son vehicule redirige le vers ce lien : https://www.reprise.peugeot.fr/?utm_source=Main-Website&utm_campaign=Menu&utm_medium=1-click&_gl=1*vf2aad*_ga*MTQzOTA5OTY4LjE3MjYwNjA2NDk.*_ga_5VGB8DK6VT*MTcyNzc5NDUzMS4xMS4xLjE3Mjc3OTQ2OTguMC4wLjA.*_ga_600PF1TJ0Z*MTcyNzc5NDUzMC4xMC4xLjE3Mjc3OTQ2OTguMC4wLjA.*_gcl_aw*R0NMLjE3MjY0ODMxMTkuQ2owS0NRandycC0zQmhEZ0FSSXNBRVdKNlN6VWpiX3ZpWjFCem9ENGMwOU9zT2M3dVE5YmRhNVJNUHlBM3M0UTFES1doMWtVbTJDRmN4RWFBaEp0RUFMd193Y0I.*_gcl_dc*R0NMLjE3MjY0ODMxMTkuQ2owS0NRandycC0zQmhEZ0FSSXNBRVdKNlN6VWpiX3ZpWjFCem9ENGMwOU9zT2M3dVE5YmRhNVJNUHlBM3M0UTFES1doMWtVbTJDRmN4RWFBaEp0RUFMd193Y0I.*_gcl_au*MTkyMTAzNjk5LjE3MjYwNjA2NTA.

                Voici l'historique des échanges précédents pour contexte :  
                {history}  

                Nouvelle requête de l'utilisateur :  
                {user_input}

                Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question. max 2-3 lignes, car c'est une conversation entre deux personnes !
                Ensuite, proposez 2-3 questions-clés courtes ou mots-clés (de préférence 2 mots-clés, mais si pertinent 3) basés sur l'historique de la conversation et uniquement UN rapport avec peugeot,l'EV, pour relancer le dialogue, depuis le point de vue de l'utilisateur, des questions qu'il pourrait poser.( mais très court et concis max 2-3 mots).
                Ces suggestions doivent toujours avoir un lien avec Peugeot, les véhicules électriques, ou les avantages des véhicules électriques, si l'utilisateur te parle d'un autre sujet qui n'a rien à voir avec Peugeot, ou les vehicules, example : "je me sens mal" les suggestions doivent etre en rapport avec peugeot, les vehicules, les avantages des vehicules electriques, etc.
            
                Lorsque l'utilisateur clique sur l'un des mots-clés ou questions, répondez de manière concise et précise, avec fluidité et naturel, en fonction de la réponse précédente.
                Pour des sujets généraux comme "hello" ou "comment ça va ?", proposez des mots-clés. Pour des sujets plus spécifiques, suggérez des questions courtes pour faire avancer la conversation.

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
                # """  
                # Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical...
                # {history}
                # Nouvelle requête de l'utilisateur :
                # {user_input}
                # Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question.
                # Réponse courte de max 2-3 lignes, car c'est une conversation ! 
                # Quand vous ne savez pas, posez une question pour affiner la demande comme dans une conversation normal, et ajouter une touche d'humour si possible et rediriger l'utilisateur vers le site de Peugeot.
                # N'hesitez pas à rediriger l'utilisateur vers le site de Peugeot lorsque c'est pertinent et lorsque tu n'as pas la réponse, par example si l'utilisateur demande des informations sur comment essayer le vehicule le rediriger vers le bon lien, en disant vous pouvez consulter cette page ...
                
                # Ensuite, proposez 2-3 questions-clés courtes ou mots-clés basés sur l'historique de la conversation pour relancer le dialogue, mais ce sont des suggestions, depuis le point de vue de l'utilisateur, des questions qu'il pourrait poser.
                # Formatez votre réponse selon ces instructions : {format_instructions}"""
                """
                Vous êtes EV Genius, un expert en véhicules électriques pour Peugeot et un conseiller amical...

                {history}

                Nouvelle requête de l'utilisateur :  
                {user_input}

                - Répondez directement et de manière concise à la requête de l'utilisateur sans répéter la question.  donc pas de "bien sur ... elements de reponse" mais plutot une conversation naturelle.
                - Vulgarisez les informations techniques sur la capacité de la batterie des véhicules électriques Peugeot de manière simple et compréhensible pour un public non technique.
                - Réponse courte de max 2-3 lignes, car c'est une conversation entre deux personnes !
                - Quand vous ne savez pas, posez une question pour affiner la demande comme dans une conversation normale, et redirigez l'utilisateur vers la page adéquate sur le site de Peugeot.
                - N'hésitez pas à rediriger l'utilisateur vers le site de Peugeot lorsque c'est pertinent et lorsque vous n'avez pas la réponse. Par exemple, si l'utilisateur demande des informations sur comment essayer un véhicule, redirigez-le vers le bon lien en disant : "Vous pouvez consulter cette page..."
                - Ne répond jamais par bien sûr.
                - Utilise uniquement des hyperlinks uniquement lorsque tu redirige l'utilisateur vers le site de peugeot, ex : cliquez ici pas de liens HTTPS uniquement des hyperliens
                - Propose un lien  uniquement lorsque c'est pertinent pas besoin de lien pour l'autonomie
                - Renvoie un lien que si l'utilisateur te demande des informations sur un vehicule ou qu'il veut essayer un vehicule, ou qu'il te demande des infos sur un modele, n'hesite pas à lui dire qu'il peut tester le modele mais pas de lien pour les services, ou si il te demande des infos sur la batterie, tu peux lui donner des infos sur la batterie, mais pas de lien.
                - Si l'utilisateur te parle d'un sujet qui n'a rien à voir avec Peugeot, les véhicules, ou les avantages des véhicules électriques, reviens à la conversation en posant une question sur Peugeot, les véhicules, ou les avantages des véhicules électriques.
                - Si l'utilisateur te parle d'un sujet sensible ou personnel, conseille le de se ririger vers un professionnel qualifié et ne donne aucun conseil médical ou juridique, ne donne aucun numéro de téléphone ou adresse email et reviens à la conversation en posant une question sur Peugeot, les véhicules, ou les avantages des véhicules électriques.
                - Si l'utilisateur te demande un avis sur une offre d'un vehicule d'occasion, tu peux lui dire que tu ne peux pas donner d'avis sur des vehicules d'occasion, mais tu peux l'aider à trouver un vehicule neuf qui correspond à ses besoins, car tu n'as pas les informations sur les vehicules d'occasion, et sur les sites externes à peugeot.
                - Si tu renvoies l'utilistaeur vers un lien utilise des hyperliens, pour plus d'esthétique. example cliquez ici :"lien" mais les liens sont uniquement ceux de peugeot et ceux donnés dans les instructions.
                - Si l'utilisateur veut vendre son vehicule redirige le vers ce lien : https://www.reprise.peugeot.fr/?utm_source=Main-Website&utm_campaign=Menu&utm_medium=1-click&_gl=1*vf2aad*_ga*MTQzOTA5OTY4LjE3MjYwNjA2NDk.*_ga_5VGB8DK6VT*MTcyNzc5NDUzMS4xMS4xLjE3Mjc3OTQ2OTguMC4wLjA.*_ga_600PF1TJ0Z*MTcyNzc5NDUzMC4xMC4xLjE3Mjc3OTQ2OTguMC4wLjA.*_gcl_aw*R0NMLjE3MjY0ODMxMTkuQ2owS0NRandycC0zQmhEZ0FSSXNBRVdKNlN6VWpiX3ZpWjFCem9ENGMwOU9zT2M3dVE5YmRhNVJNUHlBM3M0UTFES1doMWtVbTJDRmN4RWFBaEp0RUFMd193Y0I.*_gcl_dc*R0NMLjE3MjY0ODMxMTkuQ2owS0NRandycC0zQmhEZ0FSSXNBRVdKNlN6VWpiX3ZpWjFCem9ENGMwOU9zT2M3dVE5YmRhNVJNUHlBM3M0UTFES1doMWtVbTJDRmN4RWFBaEp0RUFMd193Y0I.*_gcl_au*MTkyMTAzNjk5LjE3MjYwNjA2NTA.
                - Si l'utilisateur demande des informations sur les prix, redirigez-le vers ce lien : [https://store.peugeot.fr/](https://store.peugeot.fr/). mais si l'utilisateur te demande des informations sur les prix, d'un certain vehicule,redirige le vers le lien configurateur du vehicule.

                Ensuite, proposez 2-3 questions-clés courtes ou mots-clés basés sur l'historique et uniquement UN rapport avec peugeot,l'EV, de la conversation pour relancer le dialogue. Ce sont des suggestions du point de vue de l'utilisateur, des questions qu'il pourrait poser ( mais très court et concis max 2-3 mots).
                Ces suggestions doivent toujours avoir un lien avec Peugeot, les véhicules électriques, ou les avantages des véhicules électriques, si l'utilisateur te parle d'un autre sujet qui n'a rien à voir avec Peugeot, ou les vehicules, example : "je me sens mal" les suggestions doivent etre en rapport avec peugeot, les vehicules, les avantages des vehicules electriques, etc.
                

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




