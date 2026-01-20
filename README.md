# 🏦 Implémentation d'un Modèle de Scoring Crédit - "Prêt à dépenser"

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-005571)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![CI/CD](https://github.com/Ruth972/P7_Scoring_Credit/actions/workflows/main.yml/badge.svg)

## 📄 Contexte du Projet

L'entreprise financière **"Prêt à dépenser"** souhaite mettre en œuvre un outil de **scoring crédit** pour calculer la probabilité qu'un client rembourse son crédit, puis classifier la demande en crédit accordé ou refusé.

L'objectif est de développer un algorithme de classification précis, mais surtout d'optimiser la décision métier en minimisant le coût financier des erreurs (notamment les défauts de paiement non détectés).

### 🎯 Enjeux Principaux
* **Déséquilibre de classe :** Seuls 8% des clients sont en défaut.
* **Coût Métier :** Un Faux Négatif (défaut non détecté) coûte 10 fois plus cher qu'un Faux Positif (refus à tort).
* **Interprétabilité :** Obligation légale d'expliquer la décision au client (SHAP).
* **Industrialisation :** Mise en production via une API et un Dashboard, avec une approche MLOps (CI/CD).

---

## 🏗️ Architecture Technique

Le projet est découpé en trois briques indépendantes :

1.  **Le Modèle (Training) :**
    * Pipeline `Scikit-learn` avec `SMOTE` (dans le pipeline) pour gérer le déséquilibre.
    * Modèle : `LightGBM` (Gradient Boosting).
    * Tracking des expériences : `MLflow`.
    * Optimisation du seuil de décision : **0.067** (pour minimiser la fonction de coût).

2.  **L'API (Backend - FastAPI) :**
    * Charge le modèle sérialisé (`model.pkl`).
    * Expose un endpoint `/predict` qui renvoie la probabilité, le seuil, et la décision finale.
    * Documentation automatique via Swagger UI.

3.  **Le Dashboard (Frontend - Streamlit) :**
    * Interface pour les chargés de clientèle.
    * Visualisation du score client et comparaison avec les autres clients.
    * Interprétabilité locale (Feature Importance) via `SHAP`.

---

## 🚀 Installation et Exécution Locale

### Prérequis
* Python 3.9+
* Git

### 1. Cloner le projet
```bash
git clone [https://github.com/Ruth972/P7_Scoring_Credit.git](https://github.com/Ruth972/P7_Scoring_Credit.git)
cd P7_Scoring_Credit
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Mac/Linux
# venv\Scripts\activate   # Sur Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer l'API (Backend)
```bash
uvicorn main:app --reload
```
L'API sera accessible sur http://127.0.0.1:8000.

### 5. Lancer le Dashboard (Frontend)
Dans un nouveau terminal :

```bash
streamlit run dashboard.py
```
Le dashboard s'ouvrira automatiquement dans votre navigateur.

---

## 📊 Performances du Modèle

Suite à l'optimisation des hyperparamètres et du seuil métier :

* **Modèle retenu :** LightGBM
* **AUC Test :** 0.775
* **Seuil optimal :** 0.067 (Priorité à la détection des risques)
* **Coût Métier :** Minimisé pour réduire les pertes financières (Faux Négatifs).

> **Note :** Une légère sur-performance sur le jeu d'entraînement a été identifiée et documentée, justifiant une régularisation accrue pour les futures itérations.

---

## ⚙️ MLOps & Industrialisation

Le projet intègre une chaîne d'intégration et de déploiement continu (CI/CD) :

* **Versioning :** Code hébergé sur GitHub.
* **Tests Automatisés (pytest) :**
    * Tests unitaires de l'API (Code 200, format JSON).
    * Vérification de la cohérence du modèle.
* **GitHub Actions :**
    * À chaque push sur la branche `main`, le workflow lance l'installation des dépendances et l'exécution des tests.
* **Déploiement Continu (CD) :**
    * Si les tests passent, Render déploie automatiquement la nouvelle version de l'API.
* **Monitoring :**
    * Analyse du Data Drift avec **Evidently** pour surveiller la stabilité des données en production.
