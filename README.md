# 📊 Projet APD – Plateforme MLOps de prédiction de l'Aide Publique au Développement

## 📖 Présentation

Projet Data Science & MLOps end-to-end consacré à l'analyse et à la prédiction des **engagements de l'Aide Publique au Développement (APD)** à partir de données ouvertes comprenant plus de **100 000 observations**.

L'objectif est de construire une chaîne complète allant de la préparation des données jusqu'au déploiement et au monitoring d'un modèle de Machine Learning.

### 🎯 Objectifs

- Analyser, nettoyer et préparer les données APD
- Prédire les **Engagements (K EUR)**
- Suivre les expérimentations et versionner les modèles avec MLflow
- Exposer le modèle via une API REST FastAPI
- Proposer une interface utilisateur avec Streamlit
- Automatiser les pipelines et l'intégration continue
- Détecter le data drift et permettre le réentraînement du modèle
- Conteneuriser et expérimenter le déploiement de la solution

### 🧰 Stack technique

**Data Science :** Python, Pandas, NumPy, Scikit-learn, SHAP  
**API & Application :** FastAPI, Streamlit  
**MLOps :** MLflow, DVC, Evidently AI  
**Infrastructure :** Docker, PostgreSQL, Supabase  
**CI/CD & Versioning :** Git, GitHub Actions

### 📈 Résultat principal

Le **Random Forest Regressor** sélectionné comme modèle Champion atteint :

| Métrique | Résultat |
|---|---:|
| R² | **0.724** |
| MAE | **1 484 K EUR** |

Le modèle Champion est versionné dans **MLflow Model Registry** et utilisé automatiquement par l'API de prédiction.

---

# 🏗️ Architecture MLOps

```text
                        ┌─────────────────────┐
                        │   Données APD       │
                        │   Open Data         │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Collecte / ETL      │
                        │ Prétraitement       │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Dataset préparé     │
                        │ Versioning DVC      │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Machine Learning    │
                        │ Random Forest       │
                        └──────────┬──────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────┐
            │ MLflow Tracking & Model Registry     │
            │ Runs • Métriques • Versions          │
            │ Modèle Champion                      │
            └───────────────┬──────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ FastAPI REST API│
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Streamlit UI    │
                   │ Prédictions     │
                   └─────────────────┘

      Monitoring : Evidently AI → Détection du drift → Déclenchement du réentraînement 
      CI/CD : GitHub Actions
      Infrastructure : Docker 
      Tracking Backend : Supabase PostgreSQL
```

---

# 📂 Structure du projet

```
Agence_Dvpt/
│
├── api/              # API REST FastAPI
├── app/              # Interface utilisateur Streamlit
├── data/             # Données brutes et préparées
├── docs/             # Documentation du projet
├── etl/              # Pipeline ETL et préparation des données
├── ingestion/        # Collecte et ingestion des données
├── models/           # Entraînement et gestion des modèles ML
├── monitoring/       # Détection du data drift avec Evidently
├── notebooks/        # Notebooks d'analyse et d'expérimentation
├── preprocessing/    # Nettoyage et prétraitement
├── reports/          # Rapports et résultats générés
├── scripts/          # Scripts utilitaires
├── tests/            # Tests automatisés
├── .github/
│   └── workflows/    # Pipelines CI/CD GitHub Actions
├── Dockerfile        # Construction de l'image Docker
├── requirements.txt  # Dépendances Python
└── README.md         # Documentation principale
```

---

# 🚀 Fonctionnalités

## Prétraitement

- Nettoyage des données APD
- Préparation des variables
- Construction du dataset final

---

## Machine Learning

Le modèle utilisé est un :

**Random Forest Regressor**

Objectif :

Prédire les **Engagements (K EUR)**.

---

## MLflow

Le projet utilise MLflow pour :

- suivre les expériences
- enregistrer les paramètres
- enregistrer les métriques
- sauvegarder les artefacts
- gérer les versions des modèles
- promouvoir automatiquement le modèle **Champion**

---

## Supabase

Le backend MLflow est connecté à une base PostgreSQL hébergée sur Supabase afin de conserver :

- les runs
- les paramètres
- les métriques
- les versions des modèles

---

## API FastAPI

L'API permet :

- de vérifier l'état du service avec `/health`
- d'effectuer une prédiction avec `/predict`
- de déclencher un nouvel entraînement avec `/train`
- de charger automatiquement le modèle Champion depuis MLflow Model Registry
  
### Vérification

```
GET /health
```

Retour :

```json
{
   "status":"ok",
   "message":"API disponible"
}
```

---

### Prédiction

```
POST /predict
```

Retour :

```json
{
   "prediction": ...
}
```

Le modèle utilisé est automatiquement le modèle **Champion** enregistré dans MLflow.

---

### Entraînement

```
POST /train
```

Déclenche un nouvel entraînement du modèle et enregistre le résultat dans MLflow.

---

# 🌐 Interface Streamlit

L'application Streamlit permet :

- saisir les caractéristiques d'un projet APD
- envoyer les données à l'API
- afficher la prédiction
- afficher le modèle Champion utilisé
- afficher les métriques du modèle
- consulter l'historique des prédictions

---

# 📈 Résultats du modèle Champion

| Indicateur | Valeur |
|------------|--------|
| Modèle | Random Forest |
| Version MLflow | v32 |
| Statut | Champion |
| R² | 0.724 |
| MAE | 1484 K EUR |

---

# ⚙️ Installation

```bash
git clone https://github.com/AFolsig/Agence_Dvpt.git

cd Agence_Dvpt

python -m venv agence

source agence/bin/activate

pip install -r requirements.txt
```

---

# ▶️ Lancer l'API

```bash
uvicorn api.main:app --reload
```

Documentation Swagger :

```
http://127.0.0.1:8000/docs
```

---

# ▶️ Lancer Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

# ▶️ Lancer MLflow

```bash
mlflow ui \
--backend-store-uri $MLFLOW_TRACKING_URI \
--host 127.0.0.1 \
--port 5001 \
--allowed-hosts "*"
```

---

# 🔄 Intégration Continue

Le projet utilise **GitHub Actions**.

À chaque `git push` sur la branche **main**, le workflow CI :

- installe les dépendances
- vérifie le projet
- valide le pipeline CI/CD

---

## Monitoring & Remédiation automatique

Le projet intègre un système de monitoring basé sur Evidently AI permettant de détecter les dérives des données.

### Fonctionnalités

- Génération automatique d'un rapport HTML et JSON avec Evidently.
- Intégration du rapport dans l'interface Streamlit.
- Détection des colonnes présentant une dérive statistique.
- Seuil de remédiation personnalisé fixé à 5 % de colonnes en dérive.
- Script `monitoring/check_drift_and_retrain.py` analysant automatiquement le rapport.
- Mode simulation (par défaut) afin d'éviter tout réentraînement involontaire.
- Réentraînement automatique disponible avec :

```bash
python monitoring/check_drift_and_retrain.py --execute
```

### Résultat obtenu

- 35 colonnes analysées.
- 2 colonnes détectées en dérive.
- Part des colonnes en dérive : **5.71 %**.
- Le seuil personnalisé de **5 %** déclenche une demande de remédiation.
- Un historique des actions est enregistré dans les journaux de monitoring.

# 🔮 Perspectives d'amélioration

- Réentraînement automatique avec Airflow
- Déploiement cloud complet
- Kubernetes
- Authentification renforcée
- Tests unitaires supplémentaires

---

# 📚 Technologies utilisées
   
- Python
- Pandas
- NumPy
- Scikit-Learn
- FastAPI
- Streamlit
- MLflow
- DVC
- Evidently AI
- Supabase PostgreSQL
- Docker
- Git
- GitHub Actions
   
---

# 👥 Auteurs

Projet réalisé par :

- **Augustin FAYE**
- **Mohamed AFIRI**

---

# 📸 Aperçu de la plateforme

## 🌐 API FastAPI

L'API REST permet d'entraîner le modèle, d'effectuer des prédictions, de consulter les métriques et d'accéder à l'historique des prédictions via une documentation Swagger interactive.

![API FastAPI](docs/images/api_fastapi.png)

---

## 💻 Interface Streamlit

L'application Streamlit constitue l'interface utilisateur du projet. Elle permet d'envoyer des données à l'API FastAPI, d'obtenir une prédiction et d'afficher les informations du modèle Champion enregistré dans MLflow.

![Application Streamlit](docs/images/streamlit_prediction.png)

---

## 📈 Suivi des expériences avec MLflow

MLflow assure le suivi des expériences d'entraînement, l'enregistrement des métriques, des paramètres, des artefacts ainsi que le versionnement des modèles. Le modèle Champion est automatiquement utilisé par l'API pour les prédictions.

![MLflow](docs/images/mlflow_run.png)

---
