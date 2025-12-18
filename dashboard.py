# dashboard.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# --- ⚠️ CONFIGURATION À VÉRIFIER ---
# En local, on utilise localhost.
# SUR LE CLOUD (plus tard), remplace par : "https://ton-app-heroku.herokuapp.com/predict"
API_URL = "http://127.0.0.1:8000/predict" 

# Configuration de la page
st.set_page_config(page_title="Scoring Crédit Dashboard", layout="wide")

st.title("🏦 Dashboard d'Octroi de Crédit")
st.markdown("Outil d'aide à la décision pour les chargés de clientèle.")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    data = pd.read_csv("donnees_sample.csv")
    return data

with st.spinner("Chargement des données clients..."):
    df = load_data()

# --- BARRE LATÉRALE ---
st.sidebar.header("🔍 Sélection du dossier")
client_ids = df['SK_ID_CURR'].tolist()
selected_id = st.sidebar.selectbox("ID Client", client_ids)

# --- ANALYSE DU CLIENT ---
if st.sidebar.button("Lancer l'analyse"):
    
    # 1. Récupération des données du client
    client_row = df[df['SK_ID_CURR'] == selected_id].iloc[0]
    client_dict = client_row.to_dict()
    
    # 2. Nettoyage (On enlève les colonnes qui ne sont pas des features)
    # ⚠️ Ajoute ici d'autres colonnes à exclure si nécessaire (ex: 'SK_ID_BUREAU')
    features = {k: v for k, v in client_dict.items() if k not in ['TARGET', 'SK_ID_CURR', 'index']}
    
    # 3. Appel à l'API
    try:
        response = requests.post(API_URL, json={"features": features})
        
        if response.status_code == 200:
            result = response.json()
            score = result['score']
            decision = result['decision']
            seuil = result['threshold']
            
            # --- AFFICHAGE DES RÉSULTATS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.header(f"Décision : {decision}")
                if decision == "ACCORDÉ":
                    st.success("✅ Risque Faible")
                else:
                    st.error("❌ Risque Élevé")
            
            with col2:
                st.metric("Probabilité de Défaut", f"{score:.1%}")
                st.progress(int(score * 100))
                st.caption(f"Seuil de refus : {seuil*100}%")
            
            # Affichage des données brutes (Debug)
            with st.expander("Voir les détails du dossier"):
                st.json(features)
                
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Impossible de contacter l'API.")
        st.warning("Assurez-vous que 'main.py' est bien lancé dans un autre terminal.")