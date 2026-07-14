import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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
from datetime import datetime

DATA_PATH = "data/processed/apd_ml_ready.csv"
MODEL_PATH = "models/modele_regression_rf.pkl"
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def train_model():
   print(">>> train_model() est exécutée <<<")
   load_dotenv()
   DATABASE_URL = os.getenv("DATABASE_URL")

   engine = create_engine(DATABASE_URL)
   df = pd.read_sql('SELECT * FROM "donnees"', engine)

   target = "Engagements (K EUR)"
   df = df.dropna(subset=[target])
   df[target] = pd.to_numeric(df[target], errors="coerce")
   df = df.dropna(subset=[target])
   y = df[target]
   categorical_features = [
       "Agence",
       "Nature de l'activite",
       "Pays beneficiaire",
       "Secteur",
       "Type de financement",
       "Canal de transfert"
   ]

   numeric_features = [
       "Genre",
       "nb_ODD"
   ]
   features = categorical_features + numeric_features

   for col in categorical_features:
       df[col] = df[col].fillna("Non renseigné").astype(str)

   for col in numeric_features:
       df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
   
   for col in categorical_features:
       if col in df.columns:
           df[col] = df[col].fillna("Inconnu").astype(str)

   for col in numeric_features:
       if col in df.columns:
           df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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

   mlflow.set_experiment("APD_regression")
   with mlflow.start_run():
       import subprocess

       try:
           git_commit = subprocess.check_output(
               ["git", "rev-parse", "--short", "HEAD"]
           ).decode("utf-8").strip()
           git_message = subprocess.check_output(
               ["git", "log", "-1", "--pretty=%s"]
           ).decode("utf-8").strip()

           mlflow.set_tag("git_commit", git_commit)
           mlflow.set_tag("git_message", git_message)
           dataset_version = datetime.fromtimestamp(
               os.path.getmtime(DATA_PATH)
           ).strftime("%Y-%m-%d_%H-%M-%S")

           mlflow.set_tag("dataset_version", dataset_version)
           mlflow.set_tag("dataset_path", DATA_PATH)
           mlflow.log_artifact(DATA_PATH, artifact_path="dataset")

       except:
           pass

       
       model.fit(X_train, y_train)

       y_pred = model.predict(X_test)

       r2 = r2_score(y_test, y_pred)
       mae = mean_absolute_error(y_test, y_pred)

       metrics = {
           "r2": round(r2, 3),
           "mae": round(mae, 3)
       }

       mlflow.log_param("n_estimators", 100)
       mlflow.log_metric("r2", r2)
       mlflow.log_metric("mae", mae)

       model_name = "apd_regression_model"

       mlflow.sklearn.log_model(
           sk_model=model,
           name="model",
           registered_model_name=model_name,
           serialization_format="cloudpickle"
       )

       client = MlflowClient()

       latest_version = client.get_latest_versions(model_name)[-1].version

       best_mae = None
       try:
           champion = client.get_model_version_by_alias(model_name, "champion")
           champion_run = client.get_run(champion.run_id)
           best_mae = champion_run.data.metrics.get("mae")
       except Exception:
           pass

       if True:
           client.set_registered_model_alias(
               name=model_name,
               alias="champion",
               version=latest_version
           )
           mlflow.set_tag("promotion", "champion")
           promoted = True
       else:
           mlflow.set_tag("promotion", "not_promoted")
           promoted = False

       os.makedirs("models", exist_ok=True)
       joblib.dump(model, MODEL_PATH)

       mlflow.log_artifact(MODEL_PATH)

       metrics["version"] = latest_version
       metrics["promoted"] = promoted
       metrics["best_mae_before"] = best_mae

       return metrics

       
if __name__ == "__main__":
    metrics = train_model()
    print("Entraînement terminé.")
    print(f"Métriques : {metrics}")
       
