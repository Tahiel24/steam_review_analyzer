import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentClassifier:
    """
    Clasificador en producción que consume el modelo fine-tuneado en Fase 3
    o recurre al checkpoint base si no existen los binarios locales.
    """
    def __init__(self, model_path: str = "./models/distilbert_sentiment"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Respaldo si no se descargaron o versionaron los binarios locales
        if not os.path.exists(model_path):
            print(f"Aviso: No se encontró '{model_path}'. Usando 'distilbert-base-uncased' de respaldo.")
            model_path = "distilbert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.model.to(self.device)
        self.model.eval()

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