# Testes de unidade engenharia de features: src/energy_mlops/data/build_features.py
import numpy as np
import pandas as pd

from energy_mlops.data.build_features import generate_wind_and_time_features


def test_generate_wind_and_time_features_physical_and_cyclical():
    """Valida os cálculos de derivadas físicas e variáveis cíclicas de tempo."""
    df_raw = pd.DataFrame({
        "date": ["2026-09-24 00:00:00", "2026-09-24 06:00:00"],
        "wind_speed_100m": [10.0, 20.0],
        "temperature_2m": [24.9, 19.9]
    })

    df_out = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)

    # 1. Validação da razão vento/temperatura: 10.0 / (24.9 + 0.1) = 10 / 25 = 0.4
    assert np.isclose(df_out.loc[0, "wind_temp_ratio"], 0.4)

    # 2. Validação das variáveis cíclicas de hora:
    # Hora 0 -> sin(0) = 0.0, cos(0) = 1.0
    assert np.isclose(df_out.loc[0, "hour_sin"], 0.0, atol=1e-5)
    assert np.isclose(df_out.loc[0, "hour_cos"], 1.0, atol=1e-5)

    # Hora 6 -> sin(2*pi*6/24) = sin(pi/2) = 1.0, cos(pi/2) = 0.0
    assert np.isclose(df_out.loc[1, "hour_sin"], 1.0, atol=1e-5)
    assert np.isclose(df_out.loc[1, "hour_cos"], 0.0, atol=1e-5)


def test_generate_wind_and_time_features_capacity_merge_asof():
    """Garante que a capacidade_mw é atribuída via merge_asof histórico sem duplicações."""
    df_raw = pd.DataFrame({
        "date": ["2024-11-01 12:00:00", "2026-04-01 00:00:00"],
        "wind_speed_100m": [12.0, 15.0],
        "temperature_2m": [25.0, 25.0]
    })

    df_out = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)

    # Não pode duplicar linhas
    assert len(df_out) == 2

    # Verifica se a capacidade_mw foi preenchida
    assert "capacidade_mw" in df_out.columns
    assert not df_out["capacidade_mw"].isna().any()

    # Em 2024-11-01 a capacidade esperada no histórico é maior que 10403.3
    assert df_out.loc[0, "capacidade_mw"] >= 10403.3


def test_target_fc_clipping():
    """Garante que a taxa de geração (target_fc) seja limitada entre 0.0 e 1.0."""
    df_raw = pd.DataFrame({
        "date": ["2026-09-24 12:00:00"],
        "wind_speed_100m": [10.0],
        "temperature_2m": [25.0],
        "wind_generation_mw": [999999.0]  # Valor absurdo para forçar o clip em 1.0
    })

    df_out = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)

    assert df_out.loc[0, "target_fc"] == 1.0

