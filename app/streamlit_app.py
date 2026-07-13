import time
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
   page_title="Projet APD - MLOps",
   page_icon="🌍",
   layout="wide"
)

API_URL = "http://127.0.0.1:8000"
API_KEY = "apd-secret-key"

st.sidebar.title("🌍 Projet APD")
page = st.sidebar.radio(
   "Navigation",
   [
       "Présentation",
       "Tableau de bord MLOps",
       "Architecture MLOps",
       "Données & Supabase",
       "Prédiction",
       "Historique des prédictions",
       "Monitoring & Drift",
       "Améliorations futures",
   ],
)

if page == "Tableau de bord MLOps":

   st.title("🏠 Tableau de bord MLOps")
   st.header("Projet APD - Aide Publique au Développement")

   st.markdown("""
   Bienvenue sur le tableau de bord du projet MLOps.

   Cette interface permet de suivre l'ensemble de la chaîne de traitement,
   depuis la collecte des données APD jusqu'à la prédiction finale via l'API FastAPI
   et l'interface Streamlit.
   """)

   st.divider()

   st.subheader("📊 Indicateurs principaux")

   col1, col2, col3, col4 = st.columns(4)

   with col1:
       st.metric(
           label="📁 Données brutes",
           value="106 519",
           delta="103 colonnes"
       )

   with col2:
       st.metric(
           label="🧹 Données nettoyées",
           value="72 835",
           delta="35 variables"
       )

   with col3:
       st.metric(
           label="🤖 Modèle",
           value="Random Forest",
           delta="Champion MLflow"
       )

   with col4:
       st.metric(
           label="⚙️ Pipeline",
           value="100 %",
           delta="Opérationnel"
       )

   st.success("""
   ✅ **État du pipeline**

   Toutes les composantes principales du projet MLOps sont opérationnelles :
   collecte, stockage, prétraitement, entraînement, tracking MLflow, API FastAPI
   et interface Streamlit.
   """)

   st.divider()

   st.subheader("🔁 Chaîne MLOps du projet")

   col1, col2 = st.columns(2)

   with col1:
       st.info("""
       ### 📥 Données

       - Collecte des données APD
       - Nettoyage
       - Prétraitement
       - Stockage dans Supabase
       """)

   with col2:
       st.info("""
       ### 🤖 Machine Learning

       - Modèle Random Forest
       - Prédiction des engagements en K EUR
       - Suivi des expériences avec MLflow
       - Modèle Champion enregistré
       """)

   col3, col4 = st.columns(2)

   with col3:
       st.info("""
       ### 🚀 Déploiement

       - API FastAPI sécurisée
       - Endpoint `/predict`
       - Appel du modèle Champion
       - Temps de réponse affiché
       """)

   with col4:
       st.info("""
       ### 🖥 Interface utilisateur

       - Application Streamlit
       - Formulaire de prédiction
       - Dashboard MLOps
       - Visualisation du pipeline
       """)

   st.divider()

   st.subheader("🧭 Vue synthétique du pipeline")

   st.code("""
📁 Données APD
       │
       ▼
🗄 Supabase
       │
       ▼
⚙ Prétraitement
       │
       ▼
🤖 Random Forest
       │
       ▼
📊 MLflow
       │
       ▼
🏆 Champion
       │
       ▼
🚀 FastAPI
       │
       ▼
🖥 Streamlit
   """)

   st.success("""
   ✅ Le modèle actuellement utilisé par l'application est l'alias **Champion**
   enregistré dans **MLflow Model Registry**.
   """)

   st.divider()

   st.subheader("📌 Informations du projet")

   col1, col2, col3 = st.columns(3)

   with col1:
       st.markdown("""
       **👥 Réalisé par**

       Augustin FAYE  
       Mohamed AFIRI
       """)

   with col2:
       st.markdown("""
       **🎓 Formation**

       Machine Learning Engineer / MLOps  
       École Liora *(ex DataScientest)*
       """)

   with col3:
       st.markdown("""
       **📅 Version**

       Juillet 2026  
       Projet fil rouge MLOps
       """)

elif page == "Architecture MLOps":

   st.title("🏗️ Architecture MLOps complète")

   st.markdown("""
Cette application suit une architecture **MLOps moderne** permettant
d'automatiser toute la chaîne de traitement des données APD.

Chaque composant possède un rôle précis depuis la collecte des données
jusqu'à la prédiction finale.
""")

   st.subheader("📊 Chiffres clés du projet")

   m1, m2, m3 = st.columns(3)
   m1.metric("Lignes", "72 835")
   m2.metric("Colonnes", "35")
   m3.metric("Base", "PostgreSQL")

   m4, m5, m6 = st.columns(3)
   m4.metric("Modèle", "Random Forest")
   m5.metric("Registry", "Champion")
   m6.metric("API", "FastAPI")

   st.caption("Ces indicateurs donnent une vue rapide du projet MLOps.")
   st.divider()

   st.subheader("🧩 Schéma d'architecture MLOps")

   st.markdown("""
   Ce schéma présente les interactions principales entre les données, le modèle,
   MLflow, l’API FastAPI et l’interface Streamlit.
   """)

   mermaid_code = """
   flowchart TB

       A["📂 Données APD<br/>106 519 lignes<br/>103 colonnes"]
       B["📦 Collecte automatique<br/>Batch APD"]
       C["🗄️ Supabase<br/>PostgreSQL"]
       D["⚙️ Prétraitement<br/>Nettoyage + encodage<br/>72 835 lignes / 35 variables"]
       E["🤖 Entraînement<br/>Random Forest Regressor"]
       F["📈 MLflow<br/>Tracking des runs<br/>métriques + artefacts"]
       G["🏆 Model Registry<br/>Alias Champion"]
       H["🌐 FastAPI<br/>API sécurisée<br/>/predict"]
       I["📊 Streamlit<br/>Interface utilisateur"]
       J["👤 Utilisateur<br/>Prédiction finale"]

       A --> B --> C --> D --> E --> F --> G --> H --> I --> J

       classDef data fill:#EAF3FF,stroke:#1F4E79,stroke-width:2px,color:#0B2545;
       classDef ml fill:#EAF7EA,stroke:#2E7D32,stroke-width:2px,color:#103B16;
       classDef tracking fill:#FFF3D6,stroke:#B7791F,stroke-width:2px,color:#4A2C00;
       classDef deploy fill:#F1EAFE,stroke:#6B46C1,stroke-width:2px,color:#2D1B69;
       classDef user fill:#FFFFFF,stroke:#555555,stroke-width:2px,color:#111111;

       class A,B,C data;
       class D,E ml;
       class F,G tracking;
       class H,I deploy;
       class J user;
   """

   components.html(
       f"""
       <div class="mermaid">
       {mermaid_code}
       </div>

       <script type="module">
           import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
           mermaid.initialize({{
               startOnLoad: true,
               theme: "base",
               flowchart: {{
                   curve: "basis",
                   nodeSpacing: 45,
                   rankSpacing: 65
               }},
               themeVariables: {{
                   fontFamily: "Arial",
                   fontSize: "18px",
                   primaryBorderColor: "#1F4E79",
                   lineColor: "#333333"
               }}
           }});
       </script>
       """,
       height=850,
       scrolling=True
   )

   st.success(
       "✅ Le modèle utilisé par l'API est toujours l'alias **Champion** enregistré dans **MLflow Model Registry**."
   )
     
         
   st.divider()

   st.subheader("🌐 Endpoints FastAPI")

   st.markdown("""
   Les principaux endpoints exposés par l'API sont présentés ci-dessous.
   """)

   st.table({
       "Endpoint": [
           "/",
           "/health",
           "/collect",
           "/train",
           "/predict",
           "/pipeline",
           "/metrics",
           "/data-stats",
           "/prediction-history",
           "/docs",
       ],
       "Description": [
           "Page d'accueil de l'API",
           "Vérifie que l'API est disponible",
           "Collecte automatique des données APD",
           "Entraîne le modèle Random Forest",
           "Prédit les engagements (K EUR)",
           "Exécute un scénario complet de prédiction",
           "Récupère les métriques du modèle Champion",
           "Récupère les statistiques des données Supabase",
           "Affiche l'historique des prédictions enregistrées",
           "Documentation interactive Swagger UI",
       ],
   })

   st.divider()

   st.markdown("---")

   st.subheader("🛠️ Technologies utilisées")

   col1, col2, col3, col4 = st.columns(4)

   with col1:
       st.markdown("### 🐍 Python")
       st.caption("Langage principal")

   with col2:
       st.markdown("### ⚡ FastAPI")
       st.caption("API REST")

   with col3:
       st.markdown("### 📊 Streamlit")
       st.caption("Interface")

   with col4:
       st.markdown("### 📈 MLflow")
       st.caption("Tracking")

   col1, col2, col3, col4 = st.columns(4)

   with col1:
       st.markdown("### 🗄️ Supabase")
       st.caption("PostgreSQL")

   with col2:
       st.markdown("### 🌲 Scikit-Learn")
       st.caption("Machine Learning")

   with col3:
       st.markdown("### 🐳 Docker")
       st.caption("Déploiement")

   with col4:
       st.markdown("### 🐙 GitHub")
       st.caption("Versioning")

   st.divider()

   st.header("📦 Description des composants")

   col1, col2 = st.columns(2)

   with col1:

       st.info("""
   ### 📦 Collect
   - récupération automatique des données APD
   - exécution par batch
   - alimentation automatique de Supabase
   """)

       st.info("""
   ### 🗄️ Supabase
   - Base PostgreSQL
   - stockage centralisé
   - historique des données
   - source unique pour l'entraînement
   """)

       st.info("""
   ### ⚙️ Prétraitement
   - nettoyage
   - suppression des valeurs inutiles
   - encodage
   - préparation du DataFrame
   """)

   with col2:

       st.info("""
   ### 🤖 Machine Learning
   - Random Forest Regressor
   - prédiction des engagements (K EUR)
   - pipeline Scikit-Learn
   """)

       st.info("""
   ### 📈 MLflow
   - suivi des expériences
   - métriques
   - artefacts
   - Model Registry
   - alias Champion
   """)

       st.info("""
   ### 🌐 FastAPI
   - API sécurisée (x-api-key)
   - Collecte des données
   - Entraînement du modèle
   - Prédiction
   - Métriques et statistiques
   - Historique des prédictions
   - Documentation Swagger (`/docs`)
   """)

   st.markdown("---")

   st.success("""
   ## ✅ Pipeline entièrement automatisé

   **Collect → Supabase → Prétraitement → Random Forest → MLflow → Champion → FastAPI → Streamlit**

   L'ensemble de la chaîne MLOps est automatisé, depuis la collecte des données APD jusqu'à la prédiction finale via l'interface utilisateur. 
   """)
	
   st.info(
       "🚀 Projet MLOps APD : pipeline de Machine Learning automatisé avec suivi MLflow et déploiement FastAPI/Streamlit."
   )


elif page == "Données & Supabase":

   st.title("🗄Gestion des données & Supabase")

   try:
       stats_response = requests.get(f"{API_URL}/data-stats", timeout=30)
       stats_response.raise_for_status()
       stats = stats_response.json()

       st.success("🟢 Connexion Supabase opérationnelle")

       c1, c2, c3, c4 = st.columns(4)

       c1.metric("Données brutes", f"{stats['raw_rows']:,}".replace(",", " "), f"{stats['raw_cols']} colonnes")
       c2.metric("Données nettoyées", f"{stats['clean_rows']:,}".replace(",", " "), f"{stats['clean_cols']} colonnes")
       c3.metric("Base", stats["database"])
       c4.metric("Modèle", "Random Forest", "Regressor")

   except requests.exceptions.RequestException as e:
       st.error(f"Erreur API : {e}")
   except Exception as e:
       st.error(f"Erreur Streamlit : {e}")   

   st.markdown("""
   Cette page présente la gestion des données du projet APD : collecte, nettoyage,
   transformation, stockage dans Supabase puis utilisation pour l'entraînement du modèle.
   """)

   st.markdown("## 🟢 État de la base de données")

   col1, col2, col3 = st.columns(3)

   with col1:
       st.metric(
           label="🗄️ Base",
           value="PostgreSQL",
           delta="Supabase"
       )

   with col2:
       st.metric(
           label="📋 Table principale",
           value="donnees",
           delta="Disponible"
       )

   with col3:
       st.metric(
           label="✅ Statut",
           value="Opérationnelle",
           delta="Prête ML"
       )

   st.success("""
   ✅ **Base PostgreSQL opérationnelle**  
   ✅ **Données APD disponibles**  
   ✅ **Table principale :** `donnees`  
   ✅ **Données nettoyées et transformées**  
   ✅ **Source unique utilisée pour l'entraînement du modèle**
   """)

   st.divider()

   st.subheader("🔁 Pipeline de traitement des données")

   st.caption(
       "Ce pipeline synthétise le parcours complet des données, de la collecte jusqu'à la prédiction utilisateur."
   )

   pipeline_dot = """
   digraph {
       rankdir=TB;
       graph [bgcolor="transparent"];
       node [
           shape=box,
           style="rounded,filled",
           fontname="Helvetica",
           fontsize=16,
           margin="0.35,0.25",
           width=3.6,
           height=1.0
       ];
       edge [
           color="#111827",
           arrowsize=0.9
       ];

       A [label="📁 Données APD\\n106 519 lignes\\n103 colonnes", fillcolor="#EAF2FF"];
       B [label="⚙️ Préparation des données\\nCollecte + nettoyage + transformation", fillcolor="#EAF7EA"];
       C [label="🗄️ Supabase\\nBase PostgreSQL centrale", fillcolor="#FFF7D6"];
       D [label="🤖 Modélisation\\nRandom Forest Regressor", fillcolor="#F3E8FF"];
       E [label="📈 MLflow\\nTracking + Model Registry\\nAlias Champion", fillcolor="#FDEBD0"];
       F [label="🚀 FastAPI + Streamlit\\nAPI /predict + interface utilisateur", fillcolor="#E0F2FE"];
       G [label="👤 Utilisateur\\nPrédiction finale", fillcolor="#FFFFFF"];

       A -> B -> C -> D -> E -> F -> G;
   }
   """

   st.graphviz_chart(pipeline_dot, use_container_width=True)

   st.success(
       "✅ Les données suivent un pipeline complet : préparation, stockage PostgreSQL, entraînement, suivi MLflow, déploiement API et prédiction via Streamlit."
   )

  
          
   st.divider()

   st.subheader("🗄️ Rôle de Supabase")

   col1, col2 = st.columns(2)

   with col1:
       st.markdown("""
       ### Base PostgreSQL

       - Stockage centralisé des données APD
       - Historique des données nettoyées
       - Table principale : `donnees`
       - Source utilisée pour l'entraînement
       """)

   with col2:
       st.markdown("""
       ### Utilité MLOps

       - Remplace le fichier CSV local
       - Facilite l'automatisation
       - Rend le pipeline plus reproductible
       - Prépare le projet à un usage plus professionnel
       """)

   st.divider()

   st.subheader("⚙️ Rôle des batchs")

   st.markdown("""
   Les batchs permettent d'automatiser le traitement des données.

   Ils servent à :

   - récupérer les données APD ;
   - nettoyer et transformer les données ;
   - insérer les données dans Supabase ;
   - fournir une base propre pour l'entraînement du modèle.
   """)
   st.markdown("#### 🔄 Pipeline automatisé des batches")
   st.caption("Les traitements sont exécutés automatiquement avant l'entraînement du modèle.")
   col1, col2, col3 = st.columns([1,2,1])
   with col2:
       st.graphviz_chart("""
       digraph {
           rankdir=TB;
           node [shape=box, style="rounded,filled", fontname="Arial", fontsize=13];

           A [label="📥 Collecte\\nDonnées APD", fillcolor="#E8EEF7"];
           B [label="🧹 Nettoyage\\nValeurs inutiles", fillcolor="#EAF7EA"];
           C [label="⚙️ Transformation\\nEncodage + sélection", fillcolor="#EAF7EA"];
           D [label="🗄️ Supabase\\nPostgreSQL", fillcolor="#FFF4D6"];
           E [label="🤖 Random Forest\\nEntraînement", fillcolor="#EDE7F6"];

           A -> B -> C -> D -> E;
       }
       """)

   st.divider()

   st.subheader("✅ Qualité des données")

   col1, col2 = st.columns(2)

   with col1:
       st.markdown("""
       ### Nettoyage effectué

       - Nettoyage des données
       - Traitement des valeurs manquantes
       - Sélection des variables utiles
       - Encodage des variables catégorielles
       """)

   with col2:
       st.markdown("""
       ### Données prêtes pour le ML

       - Dataset de 72 835 lignes
       - 35 variables conservées
       - Cible définie : `Engagements (K EUR)`
       - Compatible avec le pipeline Scikit-Learn
       """)

   st.success("""
   ✅ Les données APD ont été nettoyées, transformées et préparées
   avant leur utilisation par le pipeline d'entraînement Random Forest.
   """)


elif page == "Prédiction":

   st.title("🔮 Prédiction des engagements APD")
   st.success("🟢 API FastAPI disponible")

   st.markdown("""
   Cette page permet d'envoyer les caractéristiques d'un projet APD à l'API **FastAPI**,
   puis de récupérer la prédiction du modèle **Random Forest Champion**.
   """)

   st.info("La prédiction est exprimée en **K EUR**.")

   st.markdown("""
   ### 🧠 Modèle de production

   - **Modèle** : Random Forest   
   - **Alias MLflow** : Champion  
   - **API** : FastAPI  
   - **Base de données** : PostgreSQL / Supabase  
   - **Sortie prédite** : Engagements financiers en K EUR  
   """)

   st.divider()

   st.subheader("📝 Paramètres du projet APD")

   col1, col2 = st.columns(2)

   with col1:
       agence = st.selectbox(
           "Agence",
           ["AFD", "MEAE", "Proparco", "Autre"]
       )

       nature_activite = st.selectbox(
           "Nature de l'activité",
           ["Projet", "Programme", "Aide budgétaire", "Autre"]
       )

       pays = st.selectbox(
           "Pays bénéficiaire",
           ["Sénégal", "Maroc", "Côte d'Ivoire", "Tunisie", "Autre"]
       )

       secteur = st.selectbox(
           "Secteur",
           ["Education", "Santé", "Agriculture", "Eau et assainissement", "Autre"]
       )

   with col2:
       type_financement = st.selectbox(
           "Type de financement",
           ["Don", "Prêt", "Garantie", "Autre"]
       )

       canal_transfert = st.selectbox(
           "Canal de transfert",
           ["Secteur public", "ONG", "Organisation internationale", "Autre"]
       )

       genre = st.number_input(
           "Genre",
           min_value=0,
           max_value=2,
           value=1,
           step=1
       )

       odd = st.number_input(
           "ODD",
           min_value=0,
           max_value=17,
           value=3,
           step=1
       )

   payload = {
       "Agence": agence,
       "Nature_de_l_activite": nature_activite,
       "Pays_beneficiaire": pays,
       "Secteur": secteur,
       "Type_de_financement": type_financement,
       "Canal_de_transfert": canal_transfert,
       "Genre": genre,
       "ODD": odd
   }

   st.divider()

   col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

   with col_btn2:
       lancer = st.button(
           "🚀 Lancer la prédiction",
           use_container_width=True
       )

   if lancer:

       st.subheader("🔁 Chemin de la prédiction")

       prediction_dot = """
       digraph {
           rankdir=LR;
           graph [bgcolor="transparent"];
           node [
               shape=box,
               style="rounded,filled",
               fontname="Helvetica",
               fontsize=14,
               margin="0.25,0.18",
               width=2.4,
               height=0.7
           ];
           edge [
               color="#111827",
               arrowsize=0.8
           ];

           A [label="📝 Formulaire\\nStreamlit", fillcolor="#EAF2FF"];
           B [label="🌐 API\\nFastAPI", fillcolor="#E0F2FE"];
           C [label="🤖 Modèle\\nRandom Forest Champion", fillcolor="#F3E8FF"];
           D [label="💰 Prédiction\\nK EUR", fillcolor="#EAF7EA"];

           A -> B -> C -> D;
       }
       """

       st.graphviz_chart(prediction_dot, use_container_width=True)

       st.subheader("📤 Données envoyées à l'API")

       with st.expander("Voir le JSON envoyé"):
           st.json(payload)

       try:
           start_time = time.time()

           response = requests.post(
               "http://127.0.0.1:8000/predict",
               json=payload,
               headers={"x-api-key": "apd-secret-key"},
               timeout=10
           )

           response_time = round(time.time() - start_time, 2)

           if response.status_code == 200:
               result = response.json()

               prediction = (
                   result.get("prediction_engagement_k_eur")
                   or result.get("prediction")
                   or result.get("prediction_k_eur")
                   or result.get("prediction_kEUR")
               )

               if prediction is None:
                   st.error("L'API répond, mais la clé de prédiction n'a pas été trouvée.")
                   st.json(result)
               else:
                   st.success("✅ Prédiction réalisée avec succès")

                   col1, col2 = st.columns(2)

                   with col1:
                       prediction_fr = f"{float(prediction):,.2f}".replace(",", " ").replace(".", ",")

                       st.metric(
                           label="💰 Engagement prédit",
                           value=f"{prediction_fr} K EUR"
                       )

                   with col2:
                       st.metric(
                           label="⏱ Temps de réponse API",
                           value=f"{response_time:.2f} s"
                       )


                   st.info("""
                   La prédiction est calculée par le modèle **RF Champion**
                   enregistré dans **MLflow Model Registry** et exposé via l'API **FastAPI**.
                   """)

                   try:
                       metrics_response = requests.get(f"{API_URL}/metrics", timeout=5)

                       if metrics_response.status_code == 200:
                           metrics = metrics_response.json()

                           st.markdown("### 📊 Suivi du modèle MLflow")

                           m1, m2, m3, m4 = st.columns(4)

                           with m1:
                               st.metric("Modèle", metrics.get("model", "N/A"))

                           with m2:
                               st.metric("Registry", metrics.get("registry", "N/A"))

                           with m3:
                               st.metric("Version", metrics.get("version", "N/A"))

                           with m4:
                               st.metric("R²", metrics.get("r2", "N/A"))

                           mae = metrics.get("mae")

                           if mae is not None:
                               mae_fr = f"{float(mae):,.0f}".replace(",", " ")
                               st.metric("MAE", f"{mae_fr} K EUR")
                           else:
                               st.metric("MAE", "N/A")

                   except Exception:
                       st.warning("impossible de récupérer les métriques MLflow.")

               st.markdown("""
               ### ✅ Résumé technique

               - Les paramètres utilisateur sont saisis dans **Streamlit**.
               - Les données sont envoyées à l'endpoint **/predict** de **FastAPI**.
               - L'API appelle le modèle **Random Forest Champion**.
               - Le résultat est retourné puis affiché dans l'interface.
               """)

           else:
               st.error("❌ L'API répond mais la prédiction a échoué.")
               st.code(response.text)

       except Exception as e:
           st.error("❌ Impossible de contacter l'API FastAPI.")
           st.warning("Vérifie que l'API est bien lancée avec Uvicorn.")
           st.code(str(e))

   st.divider()

   st.subheader("🎯 Conclusion de la démonstration")

   st.success("""
   Cette page valide le fonctionnement complet de la chaîne de prédiction :
   saisie utilisateur, appel API FastAPI, exécution du modèle Random Forest Champion
   et affichage du montant d'engagement prédit.
   """)

elif page == "Historique des prédictions":
   st.title("📜 Historique des prédictions")

   st.markdown("""
   Cette page affiche les dernières prédictions enregistrées dans Supabase
   après chaque appel à l'endpoint **/predict**.
   """)

   try:
       response = requests.get(
           f"{API_URL}/prediction-history?limit=20",
           timeout=10
       )

       if response.status_code == 200:
           history = response.json()

           if len(history) == 0:
               st.info("Aucune prédiction enregistrée pour le moment.")
           else:
               df_history = pd.DataFrame(history)
               st.dataframe(df_history, use_container_width=True)

       else:
           st.error("Erreur lors de la récupération de l'historique.")
           st.write(response.text)

   except Exception as e:
       st.error("Impossible de contacter l'API.")
       st.write(e)

  
elif page == "Présentation":

   st.title("🎓 Projet MLOps")
   st.header("Prédiction des engagements de l'Aide Publique au Développement (APD)")

   st.markdown("""
   Cette application présente un projet complet de **Machine Learning** et de **MLOps**.

   L'objectif est de prédire le montant des engagements financiers, exprimés en **K EUR**,
   à partir des caractéristiques d'un projet d'Aide Publique au Développement.
   """)

   st.info("""
   Projet fil rouge réalisé par **Augustin FAYE** et **Mohamed AFIRI**  
   Formation **Machine Learning Engineer / MLOps**  
   École **Liora** *(ex DataScientest)*
   """)
   st.success("""
   🎯 **Objectif du projet**

   Développer une plateforme MLOps permettant d'automatiser 
   la collecte, le stockage, l'entraînement, le suivi et le 
   déploiement d'un modèle de prédiction des engagements APD.
   """)

   st.divider()

   st.subheader("👥 Parties prenantes")

   col1, col2 = st.columns(2)

   with col1:
       st.markdown("""
   **🎯 Commanditaire**

   Projet pédagogique réalisé dans le cadre de la formation
   Machine Learning Engineer / MLOps de l'École Liora.

   **👤 Utilisateurs**

   Analystes et gestionnaires de projets APD.
   """)

   with col2:
       st.markdown("""
   **⚙️ Administrateurs**

   Équipe MLOps assurant le suivi,
   le réentraînement et le déploiement.

   """)
   st.divider()

   col1, col2, col3 = st.columns(3)

   with col1:
       st.metric(
           label="📊 Données APD",
           value="106 519",
           delta="103 variables"
       )

   with col2:
       st.metric(
           label="🧠 Modèle",
           value="Random Forest",
           delta="Champion MLflow"
       )

   with col3:
       st.metric(
           label="⚙️ Pipeline",
           value="100 %",
           delta="Automatisé"
       )
   st.success(
       "✅ L'ensemble de la chaîne MLOps est entièrement opérationnel : "
       "collecte des données → stockage PostgreSQL (Supabase) → entraînement → "
       "suivi MLflow → API FastAPI → interface Streamlit."
   )
   
   st.subheader("🎯 Objectifs MLOps")

   st.markdown("""
   - Construire un pipeline **Machine Learning** complet.
   - Automatiser le traitement des données APD.
   - Centraliser les données dans **Supabase**.
   - Suivre les expériences avec **MLflow**.
   - Déployer une API **FastAPI**.
   - Offrir une interface utilisateur avec **Streamlit**.
   """)
   
   st.subheader("❓ Problématique")

   st.info("""
   Comment prédire le montant des engagements financiers (K EUR) d'un projet d'Aide
   Publique au Développement à partir de ses caractéristiques afin d'aider à la prise de décision ?
   """)

   st.subheader("🛠️Environnement technique")

   col1, col2 = st.columns(2)

   with col1:
       st.markdown("""
       **Machine Learning**

       - Python
       - Pandas
       - Scikit-Learn
       - Random Forest
       """)

   with col2:
       st.markdown("""
       **MLOps**

       - PostgreSQL (Supabase)
       - MLflow
       - FastAPI
       - GitHub Actions
       - Streamlit
       """)
   st.subheader("💻 Supports du projet")

   st.success("""
   L'application repose sur :

   - 🌐 une interface **Streamlit** destinée aux utilisateurs ;
   - ⚙️ une **API FastAPI** pour les prédictions ;
   - 🗄️ une base **PostgreSQL (Supabase)** ;
   - 📊 **MLflow** pour le suivi des expérimentations.
   """)

elif page == "Monitoring & Drift":

   st.title("📊 Monitoring & détection du drift")

   st.markdown("""
   Cette page présente le suivi de la dérive des données du projet APD.

   Le jeu de données de référence est comparé aux données courantes afin de détecter
   un changement de distribution susceptible de dégrader les performances du modèle.
   """)

   st.divider()

   st.subheader("🔍 État du monitoring")

   col1, col2, col3 = st.columns(3)

   with col1:
       st.metric(
           label="Données de référence",
           value="72 835 lignes"
       )

   with col2:
       st.metric(
           label="Données courantes",
           value="72 835 lignes"
       )

   with col3:
       st.metric(
           label="Outil",
           value="Evidently"
       )

   st.divider()

   st.subheader("📄 Rapport Evidently")

   report_path = "reports/data_drift_report.html"

   try:
       with open(report_path, "r", encoding="utf-8") as report_file:
           report_html = report_file.read()

       components.html(
           report_html,
           height=900,
           scrolling=True
       )

       st.success("✅ Rapport de drift chargé avec succès.")

   except FileNotFoundError:
       st.error(
           "Le rapport Evidently est introuvable. "
           "Exécutez d’abord : python monitoring/drift_report.py"
       )

   st.divider()

   st.subheader("♻️ Remédiation automatique")

   st.markdown("""
   Lorsque le drift dépasse le seuil défini :

   - le drift est détecté ;
   - un nouvel entraînement est déclenché ;
   - les métriques du nouveau modèle sont enregistrées dans MLflow ;
   - le meilleur modèle peut devenir le nouvel alias **Champion**.
   """)

   st.info(
       "Script utilisé : monitoring/check_drift_and_retrain.py"
   )

elif page == "Améliorations futures":

   st.title("🚀 Améliorations futures du projet")

   st.markdown("""
   Cette page présente les évolutions qui auraient pu être ajoutées avec plus de temps
   afin de rapprocher le projet d'une architecture MLOps professionnelle complète.
   """)

   st.divider()

   st.header("⏱️ Orchestration avec Airflow")

   st.markdown("""
   **Airflow** permettrait d'automatiser toute la chaîne de traitement :

   - collecte automatique des données APD ;
   - chargement dans Supabase ;
   - préparation des données ;
   - entraînement du modèle ;
   - enregistrement des métriques dans MLflow ;
   - promotion automatique du meilleur modèle.
   """)

   st.success("Objectif : remplacer les lancements manuels par un pipeline planifié et reproductible.")

   st.divider()

   st.markdown("## 📈 Évolutions du monitoring")

   st.write("""
   Le système de monitoring basé sur Evidently détecte automatiquement les dérives des données. 
   Lorsqu'un seuil de dérive est dépassé, un réentraînement du modèle peut être déclenché 
   afin de restaurer ses performances.

   Les prochaines évolutions pourraient inclure :
   """)

   st.markdown("""
   - Alertes automatiques (email, Slack ou Teams) en cas de dérive importante.
   - Tableau de bord temps réel pour suivre l'évolution des métriques du modèle.
   - Déclenchement automatique du réentraînement via Airflow après détection d'un drift.
   - Suivi continu des performances du modèle en production.
   - Historisation avancée des rapports de monitoring et des actions de remédiation.
   """)

   st.success(
       "Objectif : rendre le monitoring entièrement automatisé et proactif tout au long du cycle de vie du modèle."
   )

   st.divider()

   st.header("☸️ Déploiement avec Kubernetes")

   st.markdown("""
   **Kubernetes** permettrait de rendre l'application plus robuste et scalable :

   - déploiement de FastAPI dans un conteneur ;
   - déploiement de Streamlit dans un autre conteneur ;
   - gestion automatique des redémarrages ;
   - montée en charge si le trafic augmente ;
   - meilleure séparation entre API, interface et base de données.
   """)

   st.success("Objectif : rendre le projet plus proche d'un environnement de production réel.")

   st.divider()

   st.header("🚀 Déploiement et industrialisation")

   st.markdown("""
   Améliorations envisageables en environnement de production :

   - gestion des secrets avec un coffre sécurisé (Vault, Secrets Manager) ;
   - authentification OAuth2/JWT à la place d'une simple clé API ;
   - déploiement continu (CD) après validation de la CI GitHub Actions ;
   - publication automatique des images Docker ;
   - validation automatique des données entrantes ;
   - alertes automatiques en cas d'échec du pipeline.
   """)

   st.divider()

   st.markdown("## 🎯 Conclusion")

   st.info("""
   Le projet met désormais en œuvre les principales briques d'un pipeline MLOps moderne :

   - collecte automatisée des données ;
   - stockage dans Supabase PostgreSQL ;
   - prétraitement et entraînement du modèle ;
   - suivi des expériences avec MLflow ;
   - Model Registry avec alias Champion ;
   - API FastAPI sécurisée ;
   - interface Streamlit ;
   - intégration continue (CI) avec GitHub Actions ;
   - monitoring des données avec Evidently ;
   - détection automatique du drift ;
   - génération de rapports HTML et JSON ;
   - mécanisme de remédiation avec possibilité de déclencher le réentraînement du modèle ;
   - journalisation des actions de monitoring et de remédiation.

   Les évolutions proposées (Airflow, alertes automatiques, Kubernetes, déploiement cloud et automatisation complète du réentraînement) permettront de faire évoluer ce prototype vers une plateforme MLOps encore plus robuste et industrialisée.
   """)
