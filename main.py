import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Scoring Crédit",
    description="Microservice de prédiction du risque de crédit intégrant MLflow et Docker.",
    version="1.0.0"
)

# --- CONFIGURATION MLOPS ---
# Définition du chemin vers l'artefact du modèle MLflow.
# Utilisation d'un chemin relatif pour garantir la compatibilité entre l'environnement local et le conteneur Docker.
MODEL_PATH = "./mlruns/9/models/m-fbb1dcbf89cc40bcae5150d60a304d7d/artifacts"

# --- CHARGEMENT DU MODÈLE AU DÉMARRAGE ---
print(f"Initialisation : Chargement du modèle depuis {MODEL_PATH}...")
try:
    # Chargement du modèle via le flavor 'sklearn' de MLflow.
    # Cela permet de récupérer l'objet modèle original et d'utiliser ses méthodes natives (ex: predict_proba).
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print("Succès : Le modèle de scoring est chargé et prêt.")
except Exception as e:
    print(f"Erreur Critique : Échec du chargement du modèle MLflow via {MODEL_PATH}.")
    print(f"Exception : {e}")
    # Le modèle reste à None, l'API démarrera mais les endpoints de prédiction renverront une erreur gérée.
    model = None

class ClientData(BaseModel):
    """
    Modèle de données pour la validation des entrées API.
    Attend un dictionnaire 'features' contenant les variables du client.
    """
    features: dict

@app.get("/")
def health_check():
    """Endpoint de vérification de l'état du service (Health Check)."""
    return {
        "status": "API en ligne",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict")
def predict_credit_score(data: ClientData):
    """
    Endpoint principal de prédiction.
    1. Reçoit les données client.
    2. Nettoie et aligne les colonnes (remplit les manquantes par 0).
    3. Calcule la probabilité de défaut.
    """
    
    # Vérification de la disponibilité du modèle
    if not model:
        raise HTTPException(status_code=503, detail="Service indisponible : Le modèle n'est pas chargé.")
    
    try:
        # 1. Transformation des données d'entrée en DataFrame Pandas
        df = pd.DataFrame([data.features])
        
        # 2. Prétraitement initial (Suppression des ID)
        cols_techniques = ['SK_ID_CURR', 'TARGET', 'index', 'Unnamed: 0']
        df_clean = df.drop(columns=[c for c in cols_techniques if c in df.columns], errors='ignore')

        # ======================================================================
        # 🛡️ BLOC DE SÉCURITÉ : ALIGNEMENT AUTOMATIQUE DES COLONNES
        # ======================================================================
        # Ce bloc est indispensable pour que le modèle accepte des données incomplètes
        # (comme celles envoyées par le test unitaire).
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            
            # A. On identifie les colonnes manquantes
            missing_cols = set(expected_cols) - set(df_clean.columns)
            
            # B. On les remplit avec 0 (valeur neutre)
            if missing_cols:
                for c in missing_cols:
                    df_clean[c] = 0
            
            # C. On réordonne les colonnes strictement comme le modèle le veut
            df_clean = df_clean[expected_cols]
        # ======================================================================

        # 3. Inférence (Calcul du Score)
        proba_defaut = model.predict_proba(df_clean)[:, 1][0]
        
        # 4. Logique Métier (Seuil de décision optimisé)
        seuil_risque = 0.059 
        
        decision_finale = "REFUSÉ" if proba_defaut > seuil_risque else "ACCORDÉ"
        
        return {
            "score": float(proba_defaut),
            "decision": decision_finale,
            "threshold": seuil_risque,
            "model_source": "MLflow Registry"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {str(e)}")

if __name__ == "__main__":
    # Lancement du serveur (Configuration adaptée pour le déploiement Docker)
    uvicorn.run(app, host="0.0.0.0", port=8000)