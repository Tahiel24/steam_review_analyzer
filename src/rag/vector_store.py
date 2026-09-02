import os
import time
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from src.data.loader import load_clean_reviews


def get_chroma_collection(
    db_path: str = "./data/chroma_db",
    collection_name: str = "steam_reviews",
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Inicializa el cliente persistente y retorna la colección de ChromaDB
    con métrica de distancia coseno y modelo de embeddings MiniLM.
    """
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )


def index_reviews_batch(
    data_path: str = "./data/processed/reviews_en_clean.csv",
    db_path: str = "./data/chroma_db",
    n_samples: int = 10000,
    batch_size: int = 500,
    random_state: int = 42
):
    """
    Ejecuta el pipeline de indexación por lotes de la Fase 5:
    - Carga datos usando el loader común.
    - Muestra representativa balanceada (por defecto 10.000 reseñas).
    - Inserción en bloques de 500 registros usando upsert.
    """
    print(f"Cargando dataset para indexación desde {data_path}...")
    df = load_clean_reviews(data_path)

    # Mapeo de etiqueta textual como en Fase 5
    df['recommended_str'] = df['is_positive'].map({1: 'Recomendado', 0: 'No Recomendado'})

    sample_size = min(n_samples, len(df))
    sample_df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    collection = get_chroma_collection(db_path=db_path)
    total_samples = len(sample_df)
    print(f"Iniciando indexación de {total_samples:,} reseñas en bloques de {batch_size}...")

    start_time = time.time()
    for i in range(0, total_samples, batch_size):
        batch = sample_df.iloc[i:i + batch_size]

        ids = [
            str(row['review_id']) if 'review_id' in row and pd.notna(row['review_id']) else f"rev_{i + idx}"
            for idx, (_, row) in enumerate(batch.iterrows())
        ]
        documents = batch['review'].astype(str).tolist()

        metadatas = []
        for _, row in batch.iterrows():
            metadatas.append({
                "app_name": str(row.get('app_name', 'Unknown')),
                "recommended": str(row.get('recommended_str', 'Unknown')),
                "hours_played": float(round(row.get('hours_played', 0.0), 1))
            })

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    elapsed = time.time() - start_time
    print(f"Indexación completada: {collection.count():,} vectores almacenados en {elapsed:.2f} segundos.")


def search_semantic_evidence(
    collection,
    query: str,
    app_name: str = None,
    n_results: int = 3,
    filter_negative_only: bool = True
) -> dict:
    """
    Realiza la búsqueda semántica en ChromaDB aplicando filtrado híbrido:
    - filter_negative_only = True -> 'No Recomendado'
    - filter_negative_only = False -> 'Recomendado'
    - filter_negative_only = None -> Sin filtro de voto
    """
    where_conditions = []
    
    if filter_negative_only is True:
        where_conditions.append({"recommended": "No Recomendado"})
    elif filter_negative_only is False:
        where_conditions.append({"recommended": "Recomendado"})

    if app_name:
        where_conditions.append({"app_name": app_name})

    if len(where_conditions) > 1:
        where_clause = {"$and": where_conditions}
    elif len(where_conditions) == 1:
        where_clause = where_conditions[0]
    else:
        where_clause = None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )
    return results


def retrieve_formatted_evidence(
    collection,
    query: str,
    app_name: str = None,
    n_results: int = 3,
    filter_negative_only: bool = True
) -> list:
    """
    Recupera y formatea las evidencias en una lista de strings
    lista para ser inyectada al generador LLM de la Fase 6.
    """
    raw_results = search_semantic_evidence(
        collection=collection,
        query=query,
        app_name=app_name,
        n_results=n_results,
        filter_negative_only=filter_negative_only
    )

    evidence_texts = []
    if raw_results['documents'] and raw_results['documents'][0]:
        for idx, (doc, meta) in enumerate(zip(raw_results['documents'][0], raw_results['metadatas'][0])):
            hours = meta.get('hours_played', 0.0)
            game = meta.get('app_name', 'Desconocido')
            vote = meta.get('recommended', 'No Recomendado')
            evidence_texts.append(
                f"- [Evidencia {idx + 1} | Videojuego: {game} | Horas jugadas: {hours:.1f} hs | Voto: {vote}]:\n  \"{doc.strip()}\""
            )

    return evidence_texts


if __name__ == "__main__":
    # Permite regenerar la base vectorial ejecutando: python -m src.rag.vector_store
    index_reviews_batch()