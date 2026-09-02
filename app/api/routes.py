import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.data.loader import load_clean_reviews
from src.anomaly.detector import detect_game_anomalies
from src.rag.vector_store import get_chroma_collection, retrieve_formatted_evidence
from src.genai.summarizer import get_groq_client, stream_llm_report
from src.models.sentiment_classifier import SentimentClassifier

router = APIRouter()

# Carga de recursos
DATA_PATH = "./data/processed/reviews_en_clean.csv"
df_reviews = load_clean_reviews(DATA_PATH)
daily_anomalies = detect_game_anomalies(df_reviews)
collection = get_chroma_collection()
groq_client = get_groq_client()
sentiment_model = SentimentClassifier()

@router.get("/health", tags=["Sistema"])
def health_check():
    return {
        "status": "online",
        "total_reviews_dataset": len(df_reviews),
        "indexed_vectors_chroma": collection.count()
    }

@router.get("/games", tags=["Metadatos"])
def get_available_games():
    return {"games": sorted(df_reviews['app_name'].unique().tolist())}

@router.get("/analytics/anomalies", tags=["Series Temporales"])
def get_anomalies(game_name: str = Query(..., description="Nombre exacto del videojuego")):
    game_data = daily_anomalies[
        (daily_anomalies['app_name'] == game_name) & 
        (daily_anomalies['is_anomaly'])
    ]
    if game_data.empty:
        return {"game": game_name, "total_anomalies": 0, "anomalies": []}

    records = []
    for _, row in game_data.iterrows():
        records.append({
            "date": row['day'].strftime("%Y-%m-%d"),
            "total_reviews": int(row['total_reviews']),
            "negative_reviews": int(row['negative_reviews']),
            "negative_ratio": round(float(row['negative_ratio']), 4),
            "z_score": round(float(row['z_score']), 2)
        })
    return {"game": game_name, "total_anomalies": len(records), "anomalies": records}

@router.post("/predict/sentiment", tags=["Clasificacion del sentimiento de la reseña"])
def predict_review_sentiment(review_text: str = Query(..., description="Texto de la reseña a clasificar")):
    return sentiment_model.predict(review_text)

@router.get("/reports/generate-stream", tags=["Generación RAG"])
def generate_report_stream(
    game_name: str = Query(..., description="Videojuego a analizar"),
    user_query: str = Query(..., description="Pregunta técnica en lenguaje natural"),
    n_evidence: int = Query(3, ge=1, le=5)
):
    try:
        evidence_list = retrieve_formatted_evidence(
            collection=collection,
            query=user_query,
            app_name=game_name,
            n_results=n_evidence,
            filter_negative_only=True
        )
        generator = stream_llm_report(
            client=groq_client,
            user_query=user_query,
            game_name=game_name,
            evidence_texts=evidence_list
        )
        return StreamingResponse(generator, media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))