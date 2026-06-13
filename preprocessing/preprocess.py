import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv("ingestion/.env")

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

query = "SELECT * FROM donnees"

df = pd.read_sql(query, engine)

print("Dimensions avant nettoyage :", df.shape)

# Conversion date
if "Date d'engagement" in df.columns:
   df["Date d'engagement"] = pd.to_datetime(
       df["Date d'engagement"],
       errors="coerce"
   )

# Suppression doublons
df = df.drop_duplicates()

# Sauvegarde
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
   "data/processed/apd_processed.csv",
   index=False
)

print("Dimensions après nettoyage :", df.shape)
print("Fichier sauvegardé")
