import os
from typing import Generator, List
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_client() -> Groq:
    """Instancia el cliente de Groq leyendo la API Key de las variables de entorno."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No se encontró GROQ_API_KEY en el entorno o archivo .env")
    return Groq(api_key=api_key)


def stream_llm_report(
    client: Groq,
    user_query: str,
    game_name: str,
    evidence_texts: List[str],
    model_name: str = "openai/gpt-oss-20b"
) -> Generator[str, None, None]:
    """
    Genera el Resumen Ejecutivo en streaming vía Groq reproduciendo
    fielmente la configuración y el prompt de la Fase 6.
    """
    evidence_block = "\n\n".join(evidence_texts) if evidence_texts else "No se recuperaron fragmentos textuales directos."

    prompt = f"""Actúa como un Lead Data & Game Analyst de alto nivel técnico.
Tu tarea es generar un Resumen Ejecutivo claro, formal y procesable en ESPAÑOL para el equipo de desarrollo (Product Managers y Tech Leads) sobre un incidente de insatisfacción detectado en las reseñas de Steam.

### METADATOS DEL INCIDENTE
* Videojuego: {game_name}
* Consulta / Motivo de Análisis: "{user_query}"

### EVIDENCIA CUALITATIVA RECUPERADA (RESEÑAS REALES)
{evidence_block}

---
### INSTRUCCIONES DE FORMATO
Redacta el informe con la siguiente estructura concisa:
1. **Diagnóstico General**: Qué pasó y si representa un boicot artificial o un fallo legítimo del producto (considera la consistencia técnica de las quejas y las horas jugadas reflejadas en la evidencia para descartar un boicot con cuentas nuevas).
2. **Patrones Técnicos / Jugabilidad Clave**: Síntesis de las quejas recurrentes extraídas de la evidencia textual.
3. **Acciones Recomendadas**: 2 o 3 directivas técnicas concretas y priorizadas para el equipo de desarrollo.

Mantene un tono profesional, técnico y directo al grano, sin rodeos introductorios. Si se te pregunta por otra tarea o actividad fuera de las enlistadas aqui debes responder que no podes ayudar con eso"""

    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=2048,
        top_p=1,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta