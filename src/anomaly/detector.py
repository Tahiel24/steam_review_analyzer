import pandas as pd
import numpy as np

def detect_game_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la serie temporal diaria y detecta anomalías estadísticas por juego
    siguiendo la metodología de la Fase 4:
    - Agrupación por ('app_name', 'day').
    - Rolling Z-Score a 14 días (mínimo 7 días) sobre negative_ratio.
    - Calibración dinámica por juego (mediana de volumen y media de quejas).
    - Criterio objetivo: Z > 2.576, total_reviews >= median_daily_reviews y ratio > 2x media.
    """
    df_temp = df.copy()

    # 1. Asegurar marcas temporales y columna 'day'
    if 'day' not in df_temp.columns:
        if 'date' in df_temp.columns:
            df_temp['day'] = pd.to_datetime(df_temp['date']).dt.date
        elif 'timestamp_created' in df_temp.columns:
            df_temp['date'] = pd.to_datetime(df_temp['timestamp_created'], unit='s')
            df_temp['day'] = df_temp['date'].dt.date
        else:
            raise ValueError("No se encontró columna de fecha ('day', 'date' o 'timestamp_created').")

    # 2. Asegurar flag numérico de votos negativos
    if 'is_negative' not in df_temp.columns:
        if df_temp['recommended'].dtype == 'bool':
            df_temp['is_negative'] = (~df_temp['recommended']).astype(int)
        else:
            df_temp['is_negative'] = (df_temp['recommended'] == 0).astype(int)

    # 3. Agregación temporal por juego y día
    daily = df_temp.groupby(['app_name', 'day']).agg(
        total_reviews=('is_negative', 'count'),
        negative_reviews=('is_negative', 'sum')
    ).reset_index()

    daily['negative_ratio'] = daily['negative_reviews'] / daily['total_reviews']
    daily['day'] = pd.to_datetime(daily['day'])
    daily = daily.sort_values(by=['app_name', 'day']).reset_index(drop=True)

    # 4. Estadísticas móviles a 14 días (Rolling Z-Score)
    daily['rolling_mean'] = daily.groupby('app_name')['negative_ratio'].transform(
        lambda s: s.rolling(window=14, min_periods=7).mean()
    )
    daily['rolling_std'] = daily.groupby('app_name')['negative_ratio'].transform(
        lambda s: s.rolling(window=14, min_periods=7).std()
    )
    # Suavizado epsilon para prevenir división por cero
    daily['z_score'] = (daily['negative_ratio'] - daily['rolling_mean']) / (daily['rolling_std'] + 1e-6)

    # 5. Líneas de base por juego para calibración relativa
    game_baselines = daily.groupby('app_name').agg(
        median_daily_reviews=('total_reviews', 'median'),
        mean_negative_ratio=('negative_ratio', 'mean'),
        std_negative_ratio=('negative_ratio', 'std')
    ).reset_index()

    daily = daily.merge(game_baselines, on='app_name', how='left')

    # 6. Criterio estadístico formal (99% confianza Z > 2.576, volumen representativo, ratio > 2x media)
    daily['is_anomaly'] = (
        (daily['z_score'] > 2.576) &
        (daily['total_reviews'] >= daily['median_daily_reviews']) &
        (daily['negative_ratio'] > (2 * daily['mean_negative_ratio']))
    )

    return daily