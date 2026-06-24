from fastapi import FastAPI
from pydantic import BaseModel
from models.train_regression import train_model
from models.predict import predict_regression
from etl.collect_batch import collect_next_batch


app = FastAPI(
   title="API MLOps APD",
   description="API de prédiction des engagements APD",
   version="1.0"
)

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

@app.post("/train")
def train():
    metrics = train_model()
    return {
        "message": "Modèle entraîné avec succès",
        "metrics": metrics
    }

@app.post("/collect")
def collect():
    result = collect_next_batch()
    return result

@app.post("/predict")
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

    return {
        "prediction_engagement_k_eur": round(prediction, 2)
    }

@app.post("/pipeline")
def pipeline():
    collect_result = collect_next_batch()
    metrics = train_model()

    return {
        "message": "Pipeline exécuté avec succès",
        "collect": collect_result,
        "metrics": metrics
    }
