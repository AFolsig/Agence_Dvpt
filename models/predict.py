import mlflow.pyfunc
import pandas as pd

MODEL_NAME = "apd_regression_model"
MODEL_ALIAS = "champion"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = mlflow.pyfunc.load_model(model_uri)

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
   df_input = pd.DataFrame([input_data])

   for col in FEATURES:
       if col not in df_input.columns:
           if col in numeric_cols:
               df_input[col] = 0
           else:
               df_input[col] = "Non renseigné"

   for col in numeric_cols:
       df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0)

   for col in categorical_cols:
       df_input[col] = df_input[col].fillna("Non renseigné").astype(str)

   df_input = df_input[FEATURES]

   print("Colonnes envoyées :", df_input.columns.tolist())
   df_input = df_input[FEATURES]
   prediction = model.predict(df_input)[0]
   return float(prediction)
