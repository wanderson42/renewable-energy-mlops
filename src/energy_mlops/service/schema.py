from datetime import datetime
from typing import List  # noqa: UP035

from pydantic import BaseModel, Field


class RawWeatherRequest(BaseModel):
    """Contrato de entrada: Dados climáticos brutos fornecidos pelo cliente."""
    date: datetime = Field(..., description="Data e hora da previsão em UTC (ex: 2026-09-23T12:00:00Z)")
    wind_speed_100m: float = Field(..., ge=0.0, le=150.0, description="Velocidade do vento a 100m (m/s)")
    wind_direction_100m: float = Field(..., ge=0.0, le=360.0, description="Direção do vento a 100m (graus)")
    temperature_2m: float = Field(..., ge=10.0, le=45.0, description="Temperatura a 2m (°C)")


class BatchPredictionRequest(BaseModel):
    """Payload para envio de requisições em lote (ex: próximas 24 horas)."""
    predictions: List[RawWeatherRequest]  # noqa: UP006


class PredictionResponse(BaseModel):
    """Schema da resposta de inferência."""
    date: datetime
    predicted_fc: float = Field(..., description="Fator de Capacidade previsto (0.0 a 1.0)")
    predicted_mw: float | None = Field(None, description="Geração estimada em MW baseada na capacidade instalada da Bahia")