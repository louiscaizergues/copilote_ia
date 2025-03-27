import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Copilote IA - Analyse de ventes", layout="wide")

st.title("Copilote IA - Analyse de ventes automatisée")
st.write("Uploadez un fichier CSV et laissez l'IA résumer les tendances pour vous 🤖")

uploaded_file = st.file_uploader("Uploader votre fichier de ventes (.csv)", type="csv")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ Clé API Groq manquante. Ajoutez-la dans les secrets Streamlit Cloud.")
    st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Aperçu des données")
    st.dataframe(df.head())

    # 🎨 Visualisation avec Seaborn
    st.subheader("📊 Visualisation des tendances")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1], ax=ax)
        st.pyplot(fig)
    else:
        st.info("Pas assez de colonnes numériques pour afficher une visualisation.")

    summary = df.describe(include='all').to_string()

    prompt = f"""
    Tu es un analyste expert. Voici un tableau de données de ventes :

    {summary}

    Merci de :
    1. Résumer les tendances principales
    2. Détecter les anomalies
    3. Donner 3 recommandations concrètes et actionnables
    """

    if st.button("Analyser avec l'IA"):
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
