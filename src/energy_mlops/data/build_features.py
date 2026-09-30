import numpy as np
import pandas as pd


def generate_wind_and_time_features(df: pd.DataFrame, use_exogenous_lags: bool = True) -> pd.DataFrame:
    df_feat = df.copy()

    # 1. Converter para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(df_feat["date"]):
        df_feat["date"] = pd.to_datetime(df_feat["date"])

    # 2. Remover o timezone (UTC) se ele existir nos dados extraídos
    if df_feat["date"].dt.tz is not None:
        df_feat["date"] = df_feat["date"].dt.tz_localize(None)

    # 3. Forçar a mesma resolução (nanosegundos) para evitar o conflito [ns] vs [us]
    df_feat["date"] = df_feat["date"].astype("datetime64[ns]")

    df_feat.sort_values("date", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

# --- INTEGRAÇÃO DA CAPACIDADE INSTALADA (ABEEÓLICA - BAHIA) ---
    dados_capacidade = pd.DataFrame({
        'date': pd.to_datetime([
            '2022-01-01', 
            '2023-08-01', 
            '2024-10-01', 
            '2025-08-01', 
            '2025-10-01', 
            '2026-03-01'
        ]).astype("datetime64[ns]"),
        'capacidade_mw': [
            8500.0,
            9715.9,
            10403.3,
            11467.0,
            11682.8,
            11786.3
        ]
    }).set_index('date')
    
    # Reamostragem diária e interpolação linear
    capacidade_diaria = dados_capacidade.resample('D').interpolate(method='time').reset_index()
    
    # Prepara chave diária e ordena (exigência do merge_asof)
    df_feat['date_only'] = df_feat['date'].dt.normalize()
    df_feat = df_feat.sort_values('date_only').reset_index(drop=True)

    # merge_asof: busca a data exata ou o último registro histórico disponível no passado
    df_feat = pd.merge_asof(
        df_feat,
        capacidade_diaria,
        left_on='date_only',
        right_on='date',
        direction='backward',
        suffixes=('', '_drop')
    )
    
    # Limpeza de colunas auxiliares do merge
    for col in ['date_drop', 'date_only', 'date_y']:
        if col in df_feat.columns:
            df_feat.drop(columns=[col], inplace=True)
    
    # Preenchimento de bordas (bfill - primeiro valor disponível para datas anteriores a 2022 e ffill por segurança)
    df_feat['capacidade_mw'] = df_feat['capacidade_mw'].bfill().ffill()
    # --------------------------------------------------------------

    # Criação do Fator de Capacidade (Target Normalizado)
    if "wind_generation_mw" in df_feat.columns:
        df_feat['target_fc'] = df_feat['wind_generation_mw'] / df_feat['capacidade_mw']
        df_feat['target_fc'] = df_feat['target_fc'].clip(lower=0.0, upper=1.0)
    # --------------------------------------------------------------

    # Leis físicas
    #df_feat["wind_speed_cubed"] = df_feat["wind_speed_100m"] ** 3
    df_feat["wind_temp_ratio"] = df_feat["wind_speed_100m"] / (df_feat["temperature_2m"] + 0.1)

    # Cíclicas
    hour = df_feat["date"].dt.hour
    month = df_feat["date"].dt.month
    df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df_feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Lags exógenos úteis (sem vazar o target)
    if use_exogenous_lags and "wind_speed_100m" in df_feat.columns:
        df_feat["wind_speed_roll_mean_3h"] = (
            df_feat["wind_speed_100m"]
            .rolling(
                window=3,
                min_periods=1,
            )
            .mean()
        )

    return df_feat