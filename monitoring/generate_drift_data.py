import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

parser = argparse.ArgumentParser(
    description="Génération des données pour le monitoring du drift"
)

parser.add_argument(
    "--mode",
    choices=["simulation", "reel"],
    default="simulation",
    help="Choisir entre une dérive simulée et des données réelles"
)

args = parser.parse_args()
mode = args.mode

print(f"Mode de monitoring : {mode}")

# Création du dossier s'il n'existe pas
os.makedirs("reports", exist_ok=True)

# Jeu de données de référence
reference = pd.read_csv("data/processed/apd_ml_ready.csv")

if mode == "simulation":
    current = reference.copy()

else:
    # Charger les données réelles depuis PostgreSQL
    load_dotenv(dotenv_path=Path(".env"))

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL introuvable dans .env")

    engine = create_engine(database_url)

    try:
        current = pd.read_sql(
            'SELECT * FROM "donnees"',
            engine
        )
    finally:
        engine.dispose()

    print("Données PostgreSQL chargées :", current.shape)

if mode == "simulation":
    # Colonnes numériques
    numeric_columns = current.select_dtypes(
        include=["number"]
    ).columns

    # Simulation d'une dérive
    np.random.seed(42)

    for col in numeric_columns:
        current[col] = current[col] * np.random.normal(
            1.10, 0.05, len(current)
        )

    print("Dérive artificielle appliquée.")

else:
    print("Mode réel : aucune dérive artificielle appliquée.")

# Sauvegarde
reference_path = "reports/reference_data.csv"
current_path = "reports/current_data.csv"

if not os.path.exists(reference_path):
    reference.to_csv(reference_path, index=False)
    print("Référence historique créée.")
else:
    print("Référence historique conservée.")

current.to_csv(current_path, index=False)
print("Données courantes enregistrées.")

print("Reference :", reference.shape)
print("Current :", current.shape)
print("Drift dataset generated successfully.")
