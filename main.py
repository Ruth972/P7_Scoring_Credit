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
        
        # --- BLOC DE NETTOYAGE CRUCIAL ---
        # On supprime les colonnes "interdites" que le modèle ne connait pas
        # (L'ID, la Target, et l'index s'il traîne par là)
        cols_a_exclure = ['SK_ID_CURR', 'TARGET', 'index', 'Unnamed: 0']
        
        # On ne garde que les colonnes qui ne sont PAS dans la liste d'exclusion
        df_clean = df.drop(columns=[c for c in cols_a_exclure if c in df.columns], errors='ignore')
        # ---------------------------------

        # 2. Prédiction (sur df_clean et pas df)
        score = model.predict_proba(df_clean)[:, 1][0]
        
        seuil = 0.5 
        decision = "REFUSÉ" if score > seuil else "ACCORDÉ"
        
        return {
            "score": float(score),
            "decision": decision,
            "threshold": seuil
        }
    except Exception as e:
        # On renvoie l'erreur exacte pour le débogage
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)