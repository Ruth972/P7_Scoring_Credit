# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib  # <--- On utilise joblib pour charger le fichier .pkl

# Initialisation de l'API
app = FastAPI(
    title="API Scoring Crédit",
    description="API de prédiction du risque de défaut (Projet 7)",
    version="1.0"
)

# --- CHARGEMENT DU MODÈLE (FICHIER LOCAL) ---
print("⏳ Chargement du modèle...")
try:
    # On charge le fichier model.pkl qui est posé à côté du script
    model = joblib.load("model.pkl")
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur critique : Impossible de charger le modèle.")
    print(f"Détail : {e}")
    model = None

class ClientData(BaseModel):
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
        
        # 2. Prédiction
        # Attention : selon ton modèle, predict_proba peut varier.
        # Ici on suppose que c'est un classifier standard (LGBM, Sklearn)
        score = model.predict_proba(df)[:, 1][0]
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)