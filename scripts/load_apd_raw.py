import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
   raise ValueError("DATABASE_URL manquante dans le fichier .env")

csv_path = "data/raw/aide-publique-au-developpement.csv"

df = pd.read_csv(csv_path, sep=";")

print("Colonnes :", len(df.columns))
print("Lignes :", len(df))

engine = create_engine(DATABASE_URL)

df.to_sql(
   "apd_raw",
   engine,
   schema="public",
   if_exists="replace",
   index=False,
   chunksize=5000
)

print("Import terminé dans Supabase : table apd_raw")
