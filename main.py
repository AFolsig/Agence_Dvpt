from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

# Imports dynamiques des scripts
try:
    from training.training import run_training_pipeline
    from predict.predict import run_predict_pipeline
except ImportError as e:
    raise ImportError(
        f"Erreur d'importation. Assurez-vous que vos dossiers 'training' et 'predict' "
        f"contiennent bien un fichier '__init__.py'. Detail : {e}"
    )

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API MLOps - Aide Publique au Developpement (APD)",
    description="API d'inference et d'entrainement pour la prediction des engagements financiers.",
    version="1.0.0"
)

# --- Securisation et Validation des Donnees d'Entree (Schema Pydantic) ---
class PredictionInput(BaseModel):
    pays: str = Field(..., example="Senegal", description="Pays beneficiaire du projet")
    agence: str = Field(..., example="Agence Dakar", description="Agence AFD en charge")
    secteur: str = Field(..., example="Education", description="Secteur d'intervention du projet")
    projet_duree: int = Field(..., example=24, description="Duree prevue du projet en mois", ge=1)

# --- Endpoints de l'API ---

@app.get("/")
def read_root():
    """
    Endpoint d'accueil pour valider que l'API est en ligne.
    """
    return {
        "status": "online",
        "message": "API APD MLOps operationnelle. Accedez a /docs pour l'interface Swagger."
    }


@app.post("/training", tags=["Pipeline d'Entrainement"])
def trigger_training():
    """
    Declenche le pipeline d'entrainement complet (training_pipeline)
    """
    try:
        metrics = run_training_pipeline()
        return {
            "status": "success",
            "message": "Le pipeline d'entrainement s'est execute avec succes. Le modele a ete mis a jour.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'execution du pipeline d'entrainement : {str(e)}"
        )


@app.post("/predict", tags=["Pipeline d'Inference"])
def trigger_prediction(payload: PredictionInput):
    """
    Declenche le pipeline de prediction (predict_pipeline)
    """
    try:
        input_data = payload.model_dump()
        prediction_reale_k_eur = run_predict_pipeline(input_data)
        return {
            "status": "success",
            "input_data": input_data,
            "predicted_engagement_k_eur": round(prediction_reale_k_eur, 2)
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="Le fichier du modele (.joblib) est introuvable. Veuillez lancer l'endpoint /training d'abord."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'execution de l'inference : {str(e)}"
        )
