import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = "data/processed/apd_ml_ready.csv"
MODEL_PATH = "models/modele_regression_rf.pkl"

def train_model():
   df = pd.read_csv(DATA_PATH)

   target = "Engagements (K EUR)"
   y = df[target]
   features = [
   "Agence",
   "Nature de l'activite",
   "Pays beneficiaire",
   "Secteur",
   "Type de financement",
   "Canal de transfert",
   "Genre",
   "nb_ODD"
   ]

   X = df[features]

   cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
   num_cols = X.select_dtypes(include=["number"]).columns.tolist()

   preprocessing = ColumnTransformer([
       ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
       ("num", SimpleImputer(strategy="median"), num_cols)
   ])

   model = Pipeline([
       ("preprocessing", preprocessing),
       ("model", RandomForestRegressor(
           n_estimators=100,
           random_state=42,
           n_jobs=-1
       ))
   ])

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   model.fit(X_train, y_train)

   y_pred = model.predict(X_test)

   metrics = {
       "r2": round(r2_score(y_test, y_pred), 3),
       "mae": round(mean_absolute_error(y_test, y_pred), 3)
   }

   os.makedirs("models", exist_ok=True)
   joblib.dump(model, MODEL_PATH)

   return metrics

if __name__ == "__main__":
   print(train_model())
