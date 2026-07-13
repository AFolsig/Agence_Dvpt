import os
import pandas as pd
import numpy as np

# Création du dossier s'il n'existe pas
os.makedirs("reports", exist_ok=True)

# Jeu de données de référence
reference = pd.read_csv("data/processed/apd_ml_ready.csv")

# Copie des données
current = reference.copy()

# Colonnes numériques
numeric_columns = current.select_dtypes(include=["number"]).columns

# Simulation d'une dérive
np.random.seed(42)

for col in numeric_columns:
   current[col] = current[col] * np.random.normal(1.10, 0.05, len(current))

# Sauvegarde
reference.to_csv("reports/reference_data.csv", index=False)
current.to_csv("reports/current_data.csv", index=False)

print("Reference :", reference.shape)
print("Current :", current.shape)
print("Drift dataset generated successfully.")
