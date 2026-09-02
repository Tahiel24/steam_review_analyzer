import os
import pandas as pd

def run_cleaning_pipeline(
    raw_path: str = "./data/raw/steam_reviews.csv",
    output_path: str = "./data/processed/reviews_en_clean.csv",
    nrows: int = 500000
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de limpieza de la Fase 1:
    - Filtra idioma inglés y descarta reseñas sin texto.
    - Calcula la métrica 'word_count'.
    - Imputa nulos en 'author.playtime_at_review' con la mediana del subset.
    - Convierte 'timestamp_created' a datetime en la columna 'date'.
    - Persiste el resultado en 'output_path'.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo crudo en: {raw_path}")

    print(f"Cargando dataset crudo desde {raw_path} (nrows={nrows})...")
    df = pd.read_csv(raw_path, nrows=nrows)

    # 1. Filtrar solo inglés y eliminar filas con review nula
    df_en = df[(df['language'] == 'english') & (df['review'].notna())].copy()

    # 2. Calcular cantidad de palabras
    df_en['word_count'] = df_en['review'].str.split().str.len()

    # 3. Imputar nulos en playtime con la mediana de las reseñas en inglés
    median_playtime = df_en['author.playtime_at_review'].median()
    df_en['author.playtime_at_review'] = df_en['author.playtime_at_review'].fillna(median_playtime)

    # 4. Generar columna datetime a partir del timestamp UNIX
    df_en['date'] = pd.to_datetime(df_en['timestamp_created'], unit='s')

    # 5. Asegurar carpeta de destino y guardar archivo procesado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_en.to_csv(output_path, index=False)
    print(f"Limpieza completada con éxito. Archivo guardado en: {output_path} ({len(df_en):,} registros)")

    return df_en

if __name__ == "__main__":
    run_cleaning_pipeline()