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
    Genera el Resumen Ejecutivo en streaming vía Groq adaptándose
    al tipo de consulta (técnica, positiva, de balance o general).
    """
    evidence_block = "\n\n".join(evidence_texts) if evidence_texts else "No se recuperaron fragmentos textuales directos para este criterio."

    prompt = f"""Actúa como un Lead Data & Game Analyst de alto nivel técnico.
Tu tarea es generar un Resumen Ejecutivo claro, formal y procesable en ESPAÑOL para el equipo de producto y desarrollo sobre las reseñas de Steam.

### METADATOS DEL ANÁLISIS
* Videojuego: {game_name}
* Pregunta / Consulta del Analista: "{user_query}"

### EVIDENCIA CUALITATIVA RECUPERADA DE RESEÑAS REALES
{evidence_block}

---
### INSTRUCCIONES DE FORMATO
Responde estrictamente a la pregunta planteada apoyándote en la evidencia recuperada, siguiendo esta estructura:
1. **Diagnóstico General**: Respuesta directa a la consulta según lo observado en las reseñas. Si la consulta es sobre aspectos positivos y hay evidencia, destácalos; si es sobre quejas técnicas o rendimiento, evalúa la gravedad y consistencia del problema.
2. **Patrones Clave y Evidencia**: Síntesis de los temas o mecánicas recurrentes extraídos de las reseñas citadas.
3. **Recomendaciones / Conclusión**: 2 o 3 directivas concretas para el equipo de desarrollo o producto (ya sea para potenciar lo que funciona o corregir fallas reportadas).

Mantene un tono profesional, técnico y directo al grano, sin rodeos introductorios. Si se te pregunta por un tema ajeno al análisis de reseñas del videojuego, indica amablemente que solo estás capacitado para analizar feedback de jugadores."""

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