import pandas as pd
from sqlalchemy import create_engine
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

DATA_PATH = "data/processed/apd_ml_ready.csv"
MODEL_PATH = "models/modele_regression_rf.pkl"

def train_model():
   engine = create_engine("sqlite:///apd.db")
   df = pd.read_sql("SELECT * FROM apd_data", engine)

   target = "Engagements_(K_EUR)"
   df = df.dropna(subset=[target])
   df[target] = pd.to_numeric(df[target], errors="coerce")
   df = df.dropna(subset=[target])
   y = df[target]
   categorical_features = [
       "Agence",
       "Nature_de_l'activite",
       "Pays_beneficiaire",
       "Secteur",
       "Type_de_financement",
       "Canal_de_transfert"
   ]

   numeric_features = [
       "Genre",
       "ODD"
   ]
   features = categorical_features + numeric_features

   for col in categorical_features:
       df[col] = df[col].fillna("Non renseigné").astype(str)

   for col in numeric_features:
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

           mlflow.set_tag("git_commit", git_commit)
           mlflow.set_tag("dataset_version", "apd_ml_ready_2026_06_19")
           mlflow.set_tag("dataset_path", DATA_PATH)

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
           skops_trusted_types=["numpy.dtype"]
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

       if best_mae is None or mae <= best_mae:
           client.set_registered_model_alias(
               name=model_name,
               alias="champion",
               version=latest_version
           )
           mlflow.set_tag("promotion", "champion")
       else:
           mlflow.set_tag("promotion", "not_promoted")

       os.makedirs("models", exist_ok=True)
       joblib.dump(model, MODEL_PATH)

       mlflow.log_artifact(MODEL_PATH)

   return metrics

if __name__ == "__main__":
   print(train_model())
