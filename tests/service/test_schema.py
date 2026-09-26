# Testes do Pydantic (API REST): tests/service/test_schema.py
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

# Ajuste o import conforme o caminho real no seu projeto
from energy_mlops.service.schema import (
    BatchPredictionRequest,
    PredictionResponse,
    RawWeatherRequest,
)


def test_batch_prediction_request_valid():
    """Garante que um lote válido é aceito pelo Pydantic."""
    valid_payload = {
        "predictions": [
            {
                "date": "2026-09-24T12:00:00Z",
                "wind_speed_100m": 12.5,
                "wind_direction_100m": 180.0,
                "temperature_2m": 25.0
            }
        ]
    }
    request = BatchPredictionRequest(**valid_payload)
    assert len(request.predictions) == 1
    assert request.predictions[0].temperature_2m == 25.0

def test_raw_weather_request_invalid_wind_speed():
    """Garante rejeição de vento acima do limite de 150.0 m/s definido no schema."""
    with pytest.raises(ValidationError) as exc_info:
        RawWeatherRequest(
            date="2026-09-24T12:00:00Z",
            wind_speed_100m=151.0,  # Ultrapassa le=150.0
            wind_direction_100m=180.0,
            temperature_2m=25.0
        )
    assert "wind_speed_100m" in str(exc_info.value)

def test_raw_weather_request_invalid_temperature():
    """Garante rejeição de temperatura abaixo do limite de 10.0 °C definido no schema."""
    with pytest.raises(ValidationError) as exc_info:
        RawWeatherRequest(
            date="2026-09-24T12:00:00Z",
            wind_speed_100m=12.0,
            wind_direction_100m=180.0,
            temperature_2m=5.0  # Abaixo de ge=10.0
        )
    assert "temperature_2m" in str(exc_info.value)

def test_prediction_response_valid():
    """Valida a criação do schema de resposta da API."""
    response = PredictionResponse(
        date=datetime.now(UTC),
        predicted_fc=0.45,
        predicted_mw=2500.5
    )
    assert response.predicted_fc == 0.45