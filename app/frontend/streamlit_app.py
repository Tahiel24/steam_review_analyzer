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

st.markdown("---")

# Gráfico interactivo con Plotly (Volumen Semanal)
st.subheader(f"Volumen Semanal de Reseñas — {selected_game}")

import plotly.graph_objects as go

# 1. Agregación temporal por semana
game_weekly = (
    game_daily.set_index('day')
    .resample('W-Mon')
    .agg(
        total_reviews=('total_reviews', 'sum'),
        negative_reviews=('negative_reviews', 'sum')
    )
    .reset_index()
)

# 2. Figura de un solo gráfico
fig = go.Figure()

# Total de reseñas con área sombreada
fig.add_trace(
    go.Scatter(
        x=game_weekly['day'],
        y=game_weekly['total_reviews'],
        mode='lines',
        name="Total Reseñas",
        line=dict(color='#4d94ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(77, 148, 255, 0.25)',
        hovertemplate="Semana: %{x|%Y-%m-%d}<br>Total: %{y:,}<extra></extra>"
    )
)

# Reseñas negativas en barras
fig.add_trace(
    go.Bar(
        x=game_weekly['day'],
        y=game_weekly['negative_reviews'],
        name="Reseñas Negativas",
        marker_color="rgba(255, 59, 87, 0.9)",
        hovertemplate="Semana: %{x|%Y-%m-%d}<br>Negativas: %{y:,}<extra></extra>"
    )
)

# Configuración del layout
fig.update_layout(
    template="plotly_dark",
    barmode="overlay",
    height=420,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    ),
    margin=dict(l=50, r=40, t=50, b=40),
    xaxis=dict(title="Fecha"),
    yaxis=dict(title="Cantidad Semanal")
)

st.plotly_chart(fig, use_container_width=True)

# Sección RAG y Generación Ejecutiva
st.markdown("---")
st.subheader(" Asistente ejecutivo de triage")
st.caption("Escribí tu consulta en lenguaje natural. El sistema recuperará evidencia semántica y elaborara el informe.")

col_query, col_filter = st.columns([3, 1])

with col_query:
    user_query = st.text_input(
        "Consulta analítica:",
        placeholder="Ej: ¿Cuáles son los puntos fuertes que destacan los jugadores? o ¿Qué problemas de crashes reportaron?"
    )

with col_filter:
    review_filter = st.selectbox(
        "Tipo de reseñas a buscar:",
        ["Solo Quejas", "Solo Positivas", "Todas las Reseñas"]
    )

if st.button("Generar informe ejecutivo", type="primary") and user_query:
    # Mapeo del filtro según la selección
    if review_filter == "Solo Quejas":
        filter_neg = True
    elif review_filter == "Solo Positivas":
        filter_neg = False  # Recupera recomendadas
    else:
        filter_neg = None   # Sin filtro de polaridad

    with st.spinner("Recuperando evidencia semántica..."):
        evidence_list = retrieve_formatted_evidence(
            collection=collection,
            query=user_query,
            app_name=selected_game,
            n_results=3,
            filter_negative_only=filter_neg
        )

    st.markdown("### Informe Ejecutivo")
    report_container = st.empty()
    full_output = ""

    for chunk in stream_llm_report(groq_client, user_query, selected_game, evidence_list):
        full_output += chunk
        report_container.markdown(full_output + "▌")
    report_container.markdown(full_output)

    with st.expander("Ver evidencia textual recuperada de ChromaDB"):
        if evidence_list:
            for ev in evidence_list:
                st.info(ev)
        else:
            st.warning("No se encontraron fragmentos que coincidan con los filtros aplicados.")

# Sección Clasificador en Vivo (DistilBERT)
st.markdown("---")
st.subheader("Testear Clasificador de Sentimientos en Vivo")
test_text = st.text_area("Ingresá una reseña de prueba en inglés:", "Constant crashes and stuttering after the latest patch. Unplayable on PC.")

if st.button("Clasificar Sentimiento"):
    res = classifier.predict(test_text)
    if res["label"] == "Positivo":
        st.success(f"Predicción: **{res['label']}** (Confianza: {res['confidence']:.2%})")
    else:
        st.error(f"Predicción: **{res['label']}** (Confianza: {res['confidence']:.2%})")