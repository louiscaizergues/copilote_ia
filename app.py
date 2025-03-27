import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

st.set_page_config(page_title="Copilote IA - Analyse de ventes", layout="wide")

st.title("Copilote IA - Analyse de ventes augmentée (★ IA + ML)")
st.write("Uploadez un fichier CSV et laissez l'IA résumer les tendances pour vous, tout en bénéficiant d'une alerte intelligente grâce au machine learning \U0001f9ea")

uploaded_file = st.file_uploader("Uploader votre fichier de ventes (.csv)", type="csv")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ Clé API Groq manquante. Ajoutez-la dans les secrets Streamlit Cloud.")
    st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Aperçu des données")
    # Aperçu désactivé pour simplifier l'affichage
    # st.dataframe(df.head())

    # 🎨 Visualisation avec Seaborn
    st.subheader("📊 Visualisation des tendances")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1], ax=ax, s=60, color='#1f77b4')
        ax.set_title(f"Relation entre {num_cols[0]} et {num_cols[1]}", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        st.info("Pas assez de colonnes numériques pour afficher une visualisation.")

    # 📦 Extraction de features simples pour détection de risque
    ca_total = df['ca'].sum() if 'ca' in df.columns else 0
    quantite_total = df['quantite'].sum() if 'quantite' in df.columns else 0
    nb_lignes = len(df)
    features = pd.DataFrame([[ca_total, quantite_total, nb_lignes]], columns=['ca_total', 'quantite_total', 'nb_lignes'])

    # 📥 Chargement ou création du fichier historique ML
    histo_path = "historique_ml.csv"
    if os.path.exists(histo_path):
        df_histo = pd.read_csv(histo_path)
    else:
        df_histo = pd.DataFrame(columns=['ca_total', 'quantite_total', 'nb_lignes', 'performance_risquee'])

    # 🧠 Entraînement du modèle ML sur l'historique réel
    if not df_histo.empty:
        X = df_histo[['ca_total', 'quantite_total', 'nb_lignes']]
        y = df_histo['performance_risquee']
        model = RandomForestClassifier()
        model.fit(X, y)
        proba_risque = model.predict_proba(features)[0][1]
        st.subheader("🔢 Prédiction machine learning :")
        st.warning(f"Risque estimé de contre-performance : {int(proba_risque*100)} %")
    else:
        proba_risque = 0.0
        st.info("Pas encore assez d'historique pour prédire un risque. Ajoutez plus de fichiers.")

    # 🔄 Ajout à l'historique si l'utilisateur classe la perf
    st.subheader("🔄 Ajouter cette analyse à l'historique ?")
    label = st.radio("Cette performance vous semble-t-elle à risque ?", ("Non", "Oui"))
    if st.button("Ajouter à l'historique ML"):
        perf_risquee = 1 if label == "Oui" else 0
        nouvelle_ligne = pd.DataFrame([[ca_total, quantite_total, nb_lignes, perf_risquee]], columns=df_histo.columns)
        df_histo = pd.concat([df_histo, nouvelle_ligne], ignore_index=True)
        df_histo.to_csv(histo_path, index=False)
        st.success("Analyse ajoutée à l'historique. Le modèle s'améliorera au prochain chargement.")

    summary = df.describe(include='all').to_string()

    prompt = f"""
    Tu es un analyste senior spécialisé en performance commerciale et analyse de données.

    Voici un tableau de données de ventes issues d'une entreprise e-commerce. Tu dois faire une lecture critique de ces données comme si tu devais conseiller une direction générale.

    Note : le moteur de machine learning interne estime un risque de contre-performance à {int(proba_risque*100)} %.

    Ta mission :
    1. Identifier les tendances significatives (hausse, baisse, saisonnalité, produits ou canaux qui se démarquent)
    2. Détecter les anomalies ou incohérences (valeurs aberrantes, pics inhabituels, données manquantes)
    3. Mettre en évidence les leviers de croissance et les axes d'amélioration potentiels
    4. Proposer 3 recommandations **stratégiques** et **actionnables**, en t'appuyant sur l’analyse chiffrée

    Sois synthétique, professionnel et impactant. Utilise des formulations claires et orientées décision.

    Voici les données :

    {summary}
    """

    with st.spinner("Analyse en cours avec llama3-70b-8192..."):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "Tu es un expert en data business."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            st.subheader("📈 Résumé IA :")
            st.write(result)
        else:
            st.error("Erreur lors de l'appel à l'API :")
            st.code(response.json())
