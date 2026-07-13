import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = PROJECT_ROOT / "reports" / "data_drift_report.json"
TRAIN_SCRIPT = PROJECT_ROOT / "models" / "train_regression.py"
LOG_PATH = PROJECT_ROOT / "reports" / "drift_remediation_log.json"

# Le rapport affiche environ 5,71 % de colonnes en drift.
# Un seuil de 5 % permet donc de déclencher la remédiation.
DRIFT_THRESHOLD = 0.05


def search_values(data: Any, searched_keys: set[str]) -> list[Any]:
    """Recherche récursivement certaines clés dans un dictionnaire JSON."""
    values = []

    if isinstance(data, dict):
        for key, value in data.items():
            normalized_key = key.lower()

            if normalized_key in searched_keys:
                values.append(value)

            values.extend(search_values(value, searched_keys))

    elif isinstance(data, list):
        for item in data:
            values.extend(search_values(item, searched_keys))

    return values


def load_drift_report() -> dict:
    """Charge le rapport JSON généré par Evidently."""
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Rapport Evidently introuvable : {REPORT_PATH}\n"
            "Exécutez d'abord : python monitoring/drift_report.py"
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_drift_information(report: dict) -> dict:
    """Extrait les informations principales du rapport Evidently."""

    drift_share = 0.0
    drifted_columns = 0
    evidently_threshold = 0.5

    metrics = report.get("metrics", [])

    for metric in metrics:
        metric_name = str(metric.get("metric_name", ""))

        if metric_name.startswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            config = metric.get("config", {})

            if isinstance(value, dict):
                drifted_columns = int(value.get("count", 0) or 0)
                drift_share = float(value.get("share", 0.0) or 0.0)

            if isinstance(config, dict):
                evidently_threshold = float(
                    config.get("drift_share", 0.5) or 0.5
                )

            break

    # Décision globale d'Evidently :
    # drift si la part de colonnes affectées dépasse son seuil.
    dataset_drift = drift_share >= evidently_threshold

    # Règle personnalisée du projet :
    # remédiation dès que la part des colonnes en drift atteint 5 %.
    custom_drift_detected = drift_share >= DRIFT_THRESHOLD

    drift_detected = dataset_drift or custom_drift_detected

    return {
        "dataset_drift_evidently": dataset_drift,
        "drift_share": drift_share,
        "drifted_columns": drifted_columns,
        "threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
    }


def save_remediation_log(
    drift_information: dict,
    action: str,
    status: str,
) -> None:
    """Enregistre l’historique de la détection et de la remédiation."""

    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "report": str(REPORT_PATH),
        "drift_information": drift_information,
        "action": action,
        "status": status,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = []

    if LOG_PATH.exists():
        try:
            with LOG_PATH.open("r", encoding="utf-8") as file:
                existing_content = json.load(file)

            if isinstance(existing_content, list):
                history = existing_content
        except (json.JSONDecodeError, OSError):
            history = []

    history.append(log_entry)

    with LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def launch_retraining() -> None:
    """Lance le script existant d’entraînement du modèle."""

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script d'entraînement introuvable : {TRAIN_SCRIPT}"
        )

    print("\n🔄 Lancement du réentraînement du modèle...")

    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("✅ Réentraînement terminé avec succès.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Détection du drift et remédiation du modèle APD."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Lance réellement le réentraînement si un drift est détecté.",
    )

    args = parser.parse_args()

    print("🔎 Analyse du rapport Evidently...")

    report = load_drift_report()
    drift_information = extract_drift_information(report)

    drift_share_percent = drift_information["drift_share"] * 100
    threshold_percent = drift_information["threshold"] * 100

    print(f"Colonnes en drift : {drift_information['drifted_columns']}")
    print(f"Part de colonnes en drift : {drift_share_percent:.2f} %")
    print(f"Seuil de remédiation : {threshold_percent:.2f} %")

    if not drift_information["drift_detected"]:
        print("\n✅ Aucun drift nécessitant une remédiation.")

        save_remediation_log(
            drift_information,
            action="Aucune action",
            status="Drift non détecté",
        )
        return

    print("\n⚠️ Drift détecté : une remédiation est nécessaire.")

    if not args.execute:
        print(
            "ℹ️ Mode simulation : aucun entraînement n'est lancé.\n"
            "Pour lancer réellement le réentraînement :\n"
            "python monitoring/check_drift_and_retrain.py --execute"
        )

        save_remediation_log(
            drift_information,
            action="Réentraînement simulé",
            status="Drift détecté",
        )
        return

    try:
        launch_retraining()

        save_remediation_log(
            drift_information,
            action="Réentraînement exécuté",
            status="Succès",
        )

    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        save_remediation_log(
            drift_information,
            action="Réentraînement exécuté",
            status=f"Échec : {error}",
        )
        raise


if __name__ == "__main__":
    main()
