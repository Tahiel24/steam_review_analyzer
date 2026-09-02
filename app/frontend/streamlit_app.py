import sys
from pathlib import Path

# Añade la raíz del proyecto (dos niveles arriba de app/frontend) a sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from src.data.loader import load_clean_reviews
from src.anomaly.detector import detect_game_anomalies
from src.rag.vector_store import get_chroma_collection, retrieve_formatted_evidence
from src.genai.summarizer import get_groq_client, stream_llm_report
from src.models.sentiment_classifier import SentimentClassifier

load_dotenv()

st.set_page_config(
    page_title="Steam Review Intelligence",
    page_icon="🎮",
    layout="wide"
)


@st.cache_resource
def load_all_resources():
    df = load_clean_reviews("./data/processed/reviews_en_clean.csv")
    anomalies_df = detect_game_anomalies(df)
    coll = get_chroma_collection()
    groq = get_groq_client()
    clf = SentimentClassifier()
    return df, anomalies_df, coll, groq, clf


df, daily_stats, collection, groq_client, classifier = load_all_resources()

# ----------------- UI -----------------
st.title("Steam Review Intelligence & Triage System")
st.markdown("Sistema integral de clasificación profunda, detección estadística de anomalías y triage ejecutivo.")

st.sidebar.header("Selección de Videojuego")
available_games = sorted(df['app_name'].unique().tolist())
selected_game = st.sidebar.selectbox("Elegí un juego:", available_games)

# Filtro del juego seleccionado
game_daily = daily_stats[daily_stats['app_name'] == selected_game].copy()
anomalies_only = game_daily[game_daily['is_anomaly']]
total_game_reviews = len(df[df['app_name'] == selected_game])
avg_negativity = game_daily['negative_ratio'].mean()

# Métricas en columnas
col1, col2, col3 = st.columns(3)
col1.metric("Reseñas analizadas", f"{total_game_reviews:,}")
col2.metric("Tasa promedio de negatividad", f"{avg_negativity:.2%}")
col3.metric("Anomalías críticas", f"{len(anomalies_only)}")

st.markdown("---")

# Gráfico interactivo con Plotly
st.subheader(f"Serie Temporal de Reseñas Negativas — {selected_game}")
fig = px.line(
    game_daily,
    x='day',
    y='negative_ratio',
    labels={'day': 'Fecha', 'negative_ratio': 'Ratio Negativo'},
    template="plotly_dark"
)

if not anomalies_only.empty:
    fig.add_scatter(
        x=anomalies_only['day'],
        y=anomalies_only['negative_ratio'],
        mode='markers',
        marker=dict(color='crimson', size=10, symbol='x'),
        name='Anomalía Crítica (Z > 2.57)'
    )

st.plotly_chart(fig, use_container_width=True)

# Sección RAG y Generación Ejecutiva
st.markdown("---")
st.subheader("Asistente ejecutivo de triage")
st.caption("Escribí tu consulta en lenguaje natural. El sistema recuperará evidencia semántica y redactará el informe en streaming.")

user_query = st.text_input(
    "Consulta analítica:",
    placeholder="Ej: ¿Qué problemas técnicos o de rendimiento se reportaron con los últimos parches?"
)

if st.button("Generar informe ejecutivo", type="primary") and user_query:
    with st.spinner("Recuperando evidencia semántica..."):
        evidence_list = retrieve_formatted_evidence(
            collection=collection,
            query=user_query,
            app_name=selected_game,
            n_results=3,
            filter_negative_only=True
        )

    st.markdown("### Informe Ejecutivo")
    report_container = st.empty()
    full_output = ""

    for chunk in stream_llm_report(groq_client, user_query, selected_game, evidence_list):
        full_output += chunk
        report_container.markdown(full_output + "▌")
    report_container.markdown(full_output)

    with st.expander("Ver evidencia textual recuperada de ChromaDB"):
        for ev in evidence_list:
            st.info(ev)

# Sección Clasificador en Vivo (DistilBERT)
st.markdown("---")
st.subheader("Testear Clasificador de Sentimientos en Vivo")
test_text = st.text_area("Ingresá una reseña de prueba en inglés:", "The graphics are amazing, but the game crashes every time I enter combat.")

if st.button("Clasificar Sentimiento"):
    res = classifier.predict(test_text)
    if res["label"] == "Positivo":
        st.success(f"Predicción: **{res['label']}** (Confianza: {res['confidence']:.2%})")
    else:
        st.error(f"Predicción: **{res['label']}** (Confianza: {res['confidence']:.2%})")