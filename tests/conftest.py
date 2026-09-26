# Fixtures em escopo global do pytest (ex: mock de modelo ML ou de banco de dados)
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture(scope="session")
def dummy_model():
    """
    Treina um modelo RF minúsculo apenas para testes comportamentais.
    Ele aprende uma física sintética básica: Vento zero = energia zero. Vento forte = muita energia.
    O escopo 'session' garante que ele é treinado apenas uma vez e reaproveitado.
    """
    # Dados sintéticos de treino com a física correta
    X_train = pd.DataFrame({
        "wind_speed_100m": [0.0, 3.0, 7.0, 12.0, 18.0, 25.0],
        "temperature_2m": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        "wind_temp_ratio": [0.0, 0.12, 0.28, 0.48, 0.72, 1.0],
        "hour_sin": [0.0] * 6,
        "hour_cos": [1.0] * 6,
        "month_sin": [0.0] * 6,
        "month_cos": [1.0] * 6,
    })
    # Target sintético (Fator de Capacidade indo de 0 a 1)
    y_train = [0.0, 0.05, 0.3, 0.7, 0.95, 1.0]
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model