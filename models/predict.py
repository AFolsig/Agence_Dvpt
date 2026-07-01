import mlflow
import mlflow.pyfunc
import pandas as pd

MODEL_NAME = "apd_regression_model"
MODEL_ALIAS = "champion"

_model = None
_model_uri_loaded = None

def get_model():
   global _model, _model_uri_loaded

   model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

   if _model is None or _model_uri_loaded != model_uri:
       print(f"Chargement du modèle MLflow : {model_uri}")
       _model = mlflow.pyfunc.load_model(model_uri)
       _model_uri_loaded = model_uri

   return _model


FEATURES = [
   "Agence",
   "Nature_de_l'activite",
   "Pays_beneficiaire",
   "Secteur",
   "Type_de_financement",
   "Canal_de_transfert",
   "Genre",
   "ODD"
]

numeric_cols = ["Genre", "ODD"]

categorical_cols = [
   "Agence",
   "Nature_de_l'activite",
   "Pays_beneficiaire",
   "Secteur",
   "Type_de_financement",
   "Canal_de_transfert",
]


def predict_regression(input_data):
   model = get_model()

   df_input = pd.DataFrame([{
       "Agence": input_data.get("Agence", "Inconnu"),
       "Nature de l'activite": input_data.get("Nature_de_l_activite", "Inconnu"),
       "Pays beneficiaire": input_data.get("Pays_beneficiaire", "Inconnu"),
       "Secteur": input_data.get("Secteur", "Inconnu"),
       "Type de financement": input_data.get("Type_de_financement", "Inconnu"),
       "Canal de transfert": input_data.get("Canal_de_transfert", "Inconnu"),
       "Genre": input_data.get("Genre", 0),
       "nb_ODD": input_data.get("ODD", 0),
   }])

   categorical_cols = [
       "Agence",
       "Nature de l'activite",
       "Pays beneficiaire",
       "Secteur",
       "Type de financement",
       "Canal de transfert",
   ]

   numeric_cols = ["Genre", "nb_ODD"]

   for col in categorical_cols:
       df_input[col] = df_input[col].fillna("Inconnu").astype(str)

   for col in numeric_cols:
       df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0).astype(float)

   print("Colonnes envoyées :", df_input.columns.tolist())
   print("Types envoyés :")
   print(df_input.dtypes)

   prediction = model.predict(df_input)[0]

   return round(float(prediction), 2)
