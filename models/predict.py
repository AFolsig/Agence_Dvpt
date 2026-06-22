import pandas as pd
import mlflow.pyfunc

MODEL_URI = "models:/apd_regression_model@champion"

model = mlflow.pyfunc.load_model(MODEL_URI)

FEATURES = [
   "Agence",
   "Nature de l'activite",
   "Pays beneficiaire",
   "Secteur",
   "Type de financement",
   "Canal de transfert",
   "Genre",
   "nb_ODD"
]

numeric_cols = ["Genre", "nb_ODD"]

categorical_cols = [
   "Agence",
   "Nature de l'activite",
   "Pays beneficiaire",
   "Secteur",
   "Type de financement",
   "Canal de transfert",
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
   prediction = model.predict(df_input)[0]
   return float(prediction)
