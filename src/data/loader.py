import os
import pandas as pd

def load_clean_reviews(file_path: str = "./data/processed/reviews_en_clean.csv") -> pd.DataFrame:
    """
    Carga el dataset limpio generado en Fase 1 y aplica las normalizaciones
    básicas de tipos de datos empleadas en Fase 4:
    - Asegura formato datetime en 'date' y crea la columna 'day'.
    - Transforma 'recommended' en flags binarios numéricos ('is_negative', 'is_positive').
    - Convierte el tiempo de juego de minutos a horas ('hours_played').
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {file_path}")

    df = pd.read_csv(file_path)

    # 1. Normalización de fecha y día 
    if 'timestamp_created' in df.columns and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['timestamp_created'], unit='s')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    df['day'] = pd.to_datetime(df['date'].dt.date)

    # 2. Normalización de flags binarios de votos
    if df['recommended'].dtype == 'bool':
        df['is_negative'] = (~df['recommended']).astype(int)
        df['is_positive'] = df['recommended'].astype(int)
    else:
        df['is_negative'] = (df['recommended'] == 0).astype(int)
        df['is_positive'] = (df['recommended'] == 1).astype(int)

    # 3. Conversión de minutos a horas jugadas
    playtime_cols = [col for col in df.columns if 'playtime' in col.lower()]
    if playtime_cols:
        df['hours_played'] = (df[playtime_cols[0]] / 60.0).round(2)
    else:
        df['hours_played'] = 0.0

    return df