import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])   # <- le NOM de la variable, rien d'autre

df = pd.read_csv("aide-publique-au-developpement_clean.csv", sep=";")
print(df.shape)   # sanity check : si ça affiche (N, 1), le séparateur est ";" -> sep=";"

# Normaliser les noms de colonnes (évite les guillemets dans tout le SQL ensuite)
df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

# nettoyage minimal ici : types, NaN, doublons

df.to_sql(
    "donnees",
    engine,
    if_exists="replace",
    index=False,
    chunksize=1000,
    method="multi",
)
print(f"{len(df)} lignes chargées")