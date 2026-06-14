# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importation unique du script d'entraînement
try:
    from models.train_regression import train_model as run_training_pipeline
except ImportError as e:
    raise ImportError(f"Impossible d'importer le script d'entraînement. Détail : {e}")

app = FastAPI(
    title="API MLOps APD",
    description="API de prédiction des engagements APD (Version Autonome)",
    version="1.0"
)

# -----------------------------
# SCHEMA DES VARIABLES D'ENTREE
# -----------------------------
class PredictionInput(BaseModel):
    Agence: str = Field(..., example="Agence de Dakar")
    Nature_de_l_activite: str = Field(..., alias="Nature de l'activite", example="Prêt projet")
    Pays_beneficiaire: str = Field(..., alias="Pays beneficiaire", example="Sénégal")
    Secteur: str = Field(..., example="Éducation, formation professionnelle et emploi")
    Type_de_financement: str = Field(..., alias="Type de financement", example="Souverain")
    Canal_de_transfert: str = Field(..., alias="Canal de transfert", example="ONG")
    Genre: str = Field(..., example="Non ciblé")
    nb_ODD: int = Field(..., example=3)

    class Config:
        populate_by_name = True

# -----------------------------
# ROUTES DE L'API
# -----------------------------
@app.get("/")
def home():
    return {"message": "API APD opérationnelle"}

@app.post("/train")
def train():
    try:
        metrics = run_training_pipeline()
        return {
            "message": "Modèle entraîné avec succès",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # 1. Vérification et Chargement dynamique du modèle pkl
        model_path = "models/modele_regression_rf.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=400, 
                detail="Le fichier du modèle (.pkl) est introuvable. Veuillez d'abord exécuter la route /train."
            )
        
        model = joblib.load(model_path)
        FEATURES = model.feature_names_in_
        preprocessor = model.named_steps["preprocessing"]

        numeric_cols = []
        categorical_cols = []

        # Extraction des types de colonnes depuis le préprocesseur du modèle
        for name, transformer, cols in preprocessor.transformers_:
            if name in ["num", "numeric", "numerical"]:
                numeric_cols = list(cols)
            elif name in ["cat", "categorical"]:
                categorical_cols = list(cols)

        # 2. Récupération et conversion de la payload Pydantic
        input_data = data.model_dump(by_alias=True)
        df_input = pd.DataFrame([input_data])

        # 3. Application de votre logique de nettoyage originale
        for col in FEATURES:
            if col not in df_input.columns:
                if col in numeric_cols:
                    df_input[col] = 0
                else:
                    df_input[col] = "Non renseigné"

        for col in numeric_cols:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0)

        for col in categorical_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].fillna("Non renseigné").astype(str)

        # Alignement strict des colonnes avant la prédiction
        df_input = df_input[FEATURES]

        # 4. Prédiction finale
        prediction = model.predict(df_input)[0]
        
        return {
            "prediction_engagement_k_eur": round(float(prediction), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))