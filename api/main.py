import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from models.train_regression import train_model
from models.predict import predict_regression
from etl.collect_batch import collect_next_batch
from mlflow.tracking import MlflowClient
from datetime import datetime
import json

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
   raise ValueError("DATABASE_URL manquante dans le fichier .env")


app = FastAPI(
   title="API MLOps APD",
   description="API de prédiction des engagements APD",
   version="1.0"
)

API_KEY = os.getenv("API_KEY", "apd-secret-key")

def verify_api_key(x_api_key: str = Header(None)):
   if x_api_key != API_KEY:
       raise HTTPException(status_code=401, detail="Clé API invalide")

class PredictionInput(BaseModel):
   Agence: str
   Nature_de_l_activite: str
   Pays_beneficiaire: str
   Secteur: str
   Type_de_financement: str
   Canal_de_transfert: str
   Genre: float
   ODD: float

@app.get("/")
def home():
    return {"message": "API APD opérationnelle"}

@app.get("/health")
def health():
   return {
       "status": "ok",
       "message": "API disponible"
   }

@app.post("/train", dependencies=[Depends(verify_api_key)])
def train():
    metrics = train_model()
    return {
        "message": "Modèle entraîné avec succès",
        "metrics": metrics
    }

@app.post("/collect", dependencies=[Depends(verify_api_key)])
def collect():
    result = collect_next_batch()
    return result

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: PredictionInput):
    input_data = {
        "Agence": data.Agence,
        "Nature_de_l'activite": data.Nature_de_l_activite,
        "Pays_beneficiaire": data.Pays_beneficiaire,
        "Secteur": data.Secteur,
        "Type_de_financement": data.Type_de_financement,
        "Canal_de_transfert": data.Canal_de_transfert,
        "Genre": data.Genre,
        "ODD": data.ODD
    }

    prediction = predict_regression(input_data)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(
        "apd_regression_model",
        "champion"
    )

    engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        conn.execute(
        text("""
            INSERT INTO prediction_history
            (input_data, prediction_engagement_k_eur, model_name, model_version)
            VALUES
            (:input_data, :prediction, :model_name, :model_version)
        """),
            {
                "input_data": json.dumps(input_data),
                "prediction": float(prediction),
                "model_name": "Random Forest Regressor",
                "model_version": str(champion.version),
            },
        )

    return {
        "prediction_engagement_k_eur": round(prediction, 2)
    }

@app.post("/pipeline")
def pipeline(x_api_key: str = Header(None)):
   verify_api_key(x_api_key)

   input_data = {
       "Agence": "AFD",
       "Nature_de_l_activite": "Projet",
       "Pays_beneficiaire": "Sénégal",
       "Secteur": "Education",
       "Type_de_financement": "Prêt",
       "Canal_de_transfert": "Secteur public",
       "Genre": 1,
       "ODD": 4
   }

   prediction = predict_regression(input_data)

   return {
       "message": "Pipeline exécuté avec succès",
       "prediction_engagement_k_eur": prediction
   }

@app.get("/metrics")
def get_metrics():
   client = MlflowClient()
   model_name = "apd_regression_model"

   champion = client.get_model_version_by_alias(model_name, "champion")
   run = client.get_run(champion.run_id)

   metrics = run.data.metrics

   return {
       "model": "Random Forest Regressor",
       "registry": "Champion",
       "version": champion.version,
       "r2": round(metrics.get("r2", 0), 3),
       "mae": round(metrics.get("mae", 0), 3)
   }
 
@app.get("/data-stats")
def get_data_stats():
   engine = create_engine(DATABASE_URL)

   query = """
   SELECT
       (SELECT reltuples::bigint FROM pg_class WHERE relname = 'apd_raw') AS raw_rows,
       (SELECT reltuples::bigint FROM pg_class WHERE relname = 'donnees') AS clean_rows,
       (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'apd_raw') AS raw_cols,
       (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'donnees') AS clean_cols
   """

   stats = pd.read_sql(query, engine).iloc[0]

   return {
       "raw_rows": int(stats["raw_rows"]),
       "raw_cols": int(stats["raw_cols"]),
       "clean_rows": int(stats["clean_rows"]),
       "clean_cols": int(stats["clean_cols"]),
       "database": "PostgreSQL / Supabase",
       "raw_table": "apd_raw",
       "clean_table": "donnees"
   }
  
@app.get("/prediction-history")
def get_prediction_history(limit: int = 20):
   engine = create_engine(DATABASE_URL)

   query = """
   SELECT
       id,
       created_at,
       input_data,
       prediction_engagement_k_eur,
       model_name,
       model_version
   FROM prediction_history
   ORDER BY created_at DESC
   LIMIT %(limit)s
   """

   df = pd.read_sql(query, engine, params={"limit": limit})

   return df.to_dict(orient="records")
