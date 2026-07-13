from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


REPORTS_DIR = Path("reports")
REFERENCE_PATH = REPORTS_DIR / "reference_data.csv"
CURRENT_PATH = REPORTS_DIR / "current_data.csv"
HTML_REPORT_PATH = REPORTS_DIR / "data_drift_report.html"
JSON_REPORT_PATH = REPORTS_DIR / "data_drift_report.json"


def generate_drift_report() -> None:
   """Compare les données de référence et courantes avec Evidently."""

   if not REFERENCE_PATH.exists():
       raise FileNotFoundError(
           f"Fichier de référence introuvable : {REFERENCE_PATH}"
       )

   if not CURRENT_PATH.exists():
       raise FileNotFoundError(
           f"Fichier courant introuvable : {CURRENT_PATH}"
       )

   reference_data = pd.read_csv(REFERENCE_PATH)
   current_data = pd.read_csv(CURRENT_PATH)

   print(f"Reference : {reference_data.shape}")
   print(f"Current   : {current_data.shape}")

   report = Report(
       [
           DataDriftPreset()
       ]
   )

   snapshot = report.run(
       reference_data=reference_data,
       current_data=current_data,
   )

   snapshot.save_html(str(HTML_REPORT_PATH))
   snapshot.save_json(str(JSON_REPORT_PATH))

   print(f"HTML existe après sauvegarde : {HTML_REPORT_PATH.exists()}")
   print(f"JSON existe après sauvegarde : {JSON_REPORT_PATH.exists()}")

   print("HTML existe :", HTML_REPORT_PATH.exists())
   print("JSON existe :", JSON_REPORT_PATH.exists())

   print("Rapport Evidently généré avec succès.")
   print(f"Rapport HTML : {HTML_REPORT_PATH}")
   print(f"Rapport JSON : {JSON_REPORT_PATH}")


if __name__ == "__main__":
   generate_drift_report()
