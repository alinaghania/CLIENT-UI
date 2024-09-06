from pathlib import Path
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# Import des fonctions depuis utils.py
from utils import add_message_to_history, run_chain, process_input, save_results_to_csv

# Configuration de la page Streamlit
st.set_page_config(page_title="B.O.B")
st.title("EV Genius ")
st.markdown("**Expert en véhicules électriques Peugeot**", unsafe_allow_html=True)
# Charger et injecter le CSS depuis un fichier externe
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Charger le CSS à partir du fichier styles.css
load_css("style.css")



# Session state pour l'historique des chats et les métriques
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

if "metrics" not in st.session_state:
    st.session_state.metrics = []

# Charger le contexte du document
context = Path("parsed_data/peugeot_data.txt").read_text()

# Afficher l'historique des messages
for message in st.session_state.chat_history.messages:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    add_message_to_history(HumanMessage(content=user_input))
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    with st.chat_message("AI"):
        response = process_input(user_input, context)
        st.write(response)
    
    add_message_to_history(AIMessage(content=response))

# Exemples de questions
st.markdown('<div class="QUESTIONS EXAMPLES">', unsafe_allow_html=True)
questions = [
    "Quel est le prix de la recharge ?",
    "Bonjour, je souhaite savoir quelles sont les meilleures applications ?",
    "Quel est le taux de satisfaction client parmi les utilisateurs français qui sont passés aux véhicules électriques ?",
    "Pouvez-vous fournir un bref historique des véhicules électriques de Peugeot ?",
    "Quels sont les principaux facteurs influençant l'autonomie d'un véhicule électrique ?",
    "Dites-moi quel est le prix pour recharger ma e-208, puis le temps de recharge sur une borne."
]

for i, question in enumerate(questions, start=1):
    with st.expander(f"Question {i}"):
        st.write(question)

# Affichage des métriques et bouton de téléchargement dans la sidebar
with st.sidebar:
    st.header("Métriques et Export CSV")
    if len(st.session_state.metrics) > 0:
        df = pd.DataFrame(st.session_state.metrics)
        
        # Afficher les métriques
        st.dataframe(df)
        
        # Bouton de téléchargement
        csv = save_results_to_csv()
        st.download_button(
            label="Télécharger les résultats en CSV",
            data=csv,
            file_name="chat_results.csv",
            mime="text/csv",
        )
