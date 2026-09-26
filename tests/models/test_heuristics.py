# Testes de física/comportamento do modelo
import pandas as pd

from energy_mlops.data.build_features import generate_wind_and_time_features

# Colunas que o modelo espera receber após a engenharia de features
EXPECTED_FEATURES = [
    "wind_speed_100m", "temperature_2m", "wind_temp_ratio",
    "hour_sin", "hour_cos", "month_sin", "month_cos"
]

def test_heuristica_calmaria_zero_wind(dummy_model):
    """Garante que, se a velocidade do vento for zero, a geração seja aproximadamente zero."""
    df_raw = pd.DataFrame({
        "date": ["2026-09-24 12:00:00"],
        "wind_speed_100m": [0.0], # Calmaria total
        "temperature_2m": [30.0]
    })
    
    # 1. Passa pela pipeline de features
    df_features = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)
    X_test = df_features[EXPECTED_FEATURES]
    
    # 2. Faz a predição
    pred_fc = dummy_model.predict(X_test)[0]
    
    # 3. Asserção Comportamental
    # Vento 0 deve gerar praticamente 0 de fator de capacidade
    assert pred_fc <= 0.05, f"Falha heurística: Vento zero previu F.C. de {pred_fc}"


def test_heuristica_monotonicidade(dummy_model):
    """Garante que vento forte gera mais energia que vento fraco (ceteris paribus)."""
    df_raw = pd.DataFrame({
        "date": ["2026-09-24 12:00:00", "2026-09-24 12:00:00"],
        "wind_speed_100m": [5.0, 15.0], # Vento fraco vs forte
        "temperature_2m": [25.0, 25.0]  # Mesma temperatura
    })
    
    df_features = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)
    X_test = df_features[EXPECTED_FEATURES]
    
    preds = dummy_model.predict(X_test)
    pred_vento_fraco = preds[0]
    pred_vento_forte = preds[1]
    
    assert pred_vento_forte > pred_vento_fraco, \
        "Falha de Monotonicidade: Mais vento resultou em menos ou igual energia."


def test_heuristica_invariancia_temporal(dummy_model):
    """
    Garante que as mesmas condições meteorológicas no mesmo horário e mês, 
    mas em anos diferentes, gerem a exata mesma resposta (Invariância).
    """
    df_raw = pd.DataFrame({
        # Mudamos apenas o ano: 2025 vs 2026
        "date": ["2025-09-24 12:00:00", "2026-09-24 12:00:00"],
        "wind_speed_100m": [10.0, 10.0], 
        "temperature_2m": [25.0, 25.0]
    })
    
    df_features = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)
    X_test = df_features[EXPECTED_FEATURES]
    
    preds = dummy_model.predict(X_test)
    
    # A resposta para anos diferentes mas mesmo mês/hora deve ser idêntica
    assert preds[0] == preds[1], "Falha de Invariância: O modelo foi enviesado pelo ano."