import streamlit as st
import pandas as pd
import requests

st.title("Copilote IA - Analyse de ventes automatisée")
st.write("Uploadez un fichier CSV et laissez l'IA résumer les tendances pour vous 🤖")

uploaded_file = st.file_uploader("Uploader votre fichier de ventes (.csv)", type="csv")
api_key = st.text_input("Clé API Groq", type="password")

if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)
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
                st.subheader("📊 Résumé IA :")
                st.write(result)
            else:
                st.error("Erreur API :")
                st.code(response.json())
