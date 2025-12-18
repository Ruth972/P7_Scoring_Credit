# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

# --- ⚠️ CONFIGURATION À VÉRIFIER ---
# Remplace par le nom EXACT de ton modèle enregistré dans MLflow
MODEL_NAME = "LightGBM"  # <--- Vérifie ce nom dans ton onglet MLflow Models
VERSION = "latest"       # <--- Charge la dernière version (Production)

# Initialisation de l'API
app = FastAPI(
    title="API Scoring Crédit",
    description="API de prédiction du risque de défaut (Projet 7)",
    version="1.0"
)

# --- CHARGEMENT DU MODÈLE ---
print("⏳ Chargement du modèle...")
try:
    # Charge le modèle depuis le dossier mlruns local
    model_uri = f"models:/{MODEL_NAME}/{VERSION}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modèle '{MODEL_NAME}' chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur critique : Impossible de charger le modèle.")
    print(f"Détail : {e}")
    model = None

# --- FORMAT DES DONNÉES ---
class ClientData(BaseModel):
    # On attend un dictionnaire de features (ex: {"EXT_SOURCE_3": 0.5, ...})
    features: dict

@app.get("/")
def index():
    return {"message": "API Scoring Credit en ligne 🚀"}

@app.post("/predict")
def predict(data: ClientData):
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")
    
    try:
        # 1. Conversion JSON -> DataFrame
        df = pd.DataFrame([data.features])
        
        # 2. Prédiction (Probabilité de la classe 1 = Défaut)
        # predict_proba renvoie [[prob_0, prob_1]] -> on prend prob_1
        score = model.predict_proba(df)[:, 1][0]
        
        # 3. Décision Métier
        # ⚠️ Tu peux ajuster ce seuil (ex: 0.4 ou 0.6) selon ton coût métier
        seuil = 0.5 
        decision = "REFUSÉ" if score > seuil else "ACCORDÉ"
        
        return {
            "score": float(score),
            "decision": decision,
            "threshold": seuil
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Lancement local pour le développement
    uvicorn.run(app, host="0.0.0.0", port=8000)