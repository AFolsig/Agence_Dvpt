from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Imports dynamiques des scripts
try:
    from training.training import run_training_pipeline
    from predict.predict import run_predict_pipeline
except ImportError as e:
    raise ImportError(
        f"Erreur d'importation. Assurez-vous que vos dossiers 'training' et 'predict' "
        f"contiennent bien un fichier '__init__.py'. DÇtail : {e}"
    )

# Initialisation FastAPI
app = FastAPI(
    title="API MLOps - Aide Publique au DÇveloppement (APD)",
    description="API d'infÇrence et d'entraånement pour la prÇdiction des engagements financiers.",
    version="1.0.0"
)

# -----------------------------
#   SCHEMA DES VARIABLES D'ENTRêE
# -----------------------------
class PredictionInput(BaseModel):

    # Variables catÇgorielles
    pays: str = Field(..., example="SÇnÇgal")
    agence: str = Field(..., example="Agence Dakar")
    secteur: str = Field(..., example="êducation")
    canal_transfert: str = Field(..., example="Canal bilatÇral")
    type_flux: str = Field(..., example="Pràt")
    type_financement: str = Field(..., example="Souverain")
    modalite_cooperation: str = Field(..., example="Projet")
    objet: str = Field(..., example="Construction d'infrastructures")
    priorite_cicid: str = Field(..., example="Climat")
    region: str = Field(..., example="Afrique subsaharienne")
    sous_region: str = Field(..., example="Afrique de l'Ouest")

    # Marqueurs
    marqueur_climat: int = Field(..., example=1)
    marqueur_genre: int = Field(..., example=0)
    marqueur_env: int = Field(..., example=0)

    # Variables numÇriques
    nb_odd: int = Field(..., example=2)
    principal_debourse_k_eur: float = Field(..., example=500)
    principal_restant_k_eur: float = Field(..., example=1500)
    interets_k_eur: float = Field(..., example=20)
    dividendes_k_eur: float = Field(..., example=0)
    taille_projet: float = Field(..., example=2500)
    duree_mois: int = Field(..., example=24)


# -----------------------------
#   ENDPOINTS
# -----------------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "API APD MLOps opÇrationnelle. AccÇdez Ö /docs pour Swagger."
    }


@app.post("/training", tags=["Pipeline d'Entraånement"])
def trigger_training():
    try:
        metrics = run_training_pipeline()
        return {
            "status": "success",
            "message": "Pipeline d'entraånement exÇcutÇ avec succäs.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'entraånement : {str(e)}"
        )


@app.post("/predict", tags=["Pipeline d'InfÇrence"])
def trigger_prediction(payload: PredictionInput):
    try:
        input_data = payload.model_dump()
        prediction_k_eur = run_predict_pipeline(input_data)

        return {
            "status": "success",
            "input_data": input_data,
            "predicted_engagement_k_eur": round(prediction_k_eur, 2)
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Modäle introuvable. Lancez d'abord /training."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'infÇrence : {str(e)}"
        )

