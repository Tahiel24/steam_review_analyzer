import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ruta absoluta por defecto hacia la carpeta models/distilbert_sentiment en la raíz
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "distilbert_sentiment"


class SentimentClassifier:
    """
    Clasificador en producción que consume el modelo fine-tuneado en Fase 3.
    """
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Si no se pasa ruta, usamos la ruta absoluta calculada
        target_path = Path(model_path) if model_path else DEFAULT_MODEL_DIR

        if not target_path.exists():
            print(f"[ALERTA] No se encontró el directorio de modelo en: {target_path}")
            print("Usando 'distilbert-base-uncased' de respaldo (sin calibrar).")
            load_target = "distilbert-base-uncased"
            self.model = AutoModelForSequenceClassification.from_pretrained(load_target, num_labels=2)
        else:
            print(f"[INFO] Cargando modelo entrenado desde: {target_path}")
            load_target = str(target_path)
            # Carga directa de pesos y configuración entrenada
            self.model = AutoModelForSequenceClassification.from_pretrained(load_target)

        self.tokenizer = AutoTokenizer.from_pretrained(load_target)
        self.model.to(self.device)
        self.model.eval()

        # Mapeo según Fase 3: 0 -> Negativo (No Recomendado), 1 -> Positivo (Recomendado)
        self.label_map = {0: "Negativo", 1: "Positivo"}

    def predict(self, text: str) -> dict:
        """Predice la polaridad y confianza para un texto individual."""
        if not isinstance(text, str) or not text.strip():
            return {"label": "Desconocido", "sentiment_id": -1, "confidence": 0.0}

        inputs = self.tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            sentiment_id = int(torch.argmax(probabilities).item())

        return {
            "label": self.label_map.get(sentiment_id, "Desconocido"),
            "sentiment_id": sentiment_id,
            "confidence": round(float(probabilities[sentiment_id].item()), 4)
        }