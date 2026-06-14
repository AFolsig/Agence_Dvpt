import pandas as pd
import joblib

model = joblib.load("models/modele_regression_rf.pkl")
FEATURES = model.feature_names_in_

preprocessor = model.named_steps["preprocessing"]

numeric_cols = []
categorical_cols = []

for name, transformer, cols in preprocessor.transformers_:
   if name in ["num", "numeric", "numerical"]:
       numeric_cols = list(cols)
   elif name in ["cat", "categorical"]:
       categorical_cols = list(cols)

def predict_regression(input_data):
   df_input = pd.DataFrame([input_data])

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

   df_input = df_input[FEATURES]

   prediction = model.predict(df_input)[0]
   return float(prediction)
