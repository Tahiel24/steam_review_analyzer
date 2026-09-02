from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.routes import router

load_dotenv()

app = FastAPI(
    title="Steam Review Intelligence API",
    description="API REST para análisis de sentimiento, detección de anomalías y triage ejecutivo RAG.",
    version="1.0.0"
)

# Registramos todas las rutas
app.include_router(router)