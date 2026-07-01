import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
   raise ValueError("DATABASE_URL manquante dans le fichier .env")

csv_path = "data/processed/apd_ml_ready.csv"

print("Lecture du dataset ML final...")
df = pd.read_csv(csv_path)

print("Dimensions du dataset ML :", df.shape)
print("Colonnes :", len(df.columns))

engine = create_engine(DATABASE_URL)

print("Écriture dans Supabase : table donnees...")
df.to_sql(
   "donnees",
   engine,
   if_exists="replace",
   index=False,
   chunksize=1000
)

print("Table donnees remplacée avec succès.")
print("Lignes :", df.shape[0])
print("Colonnes :", df.shape[1])
