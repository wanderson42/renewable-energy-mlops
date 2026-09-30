# Testes do Pandera (Extratores): src/energy_mlops/data/schema.py
import pandas as pd
import pandera as pa
import pytest

from energy_mlops.data.schema import EnergySchema, WeatherSchema


def test_weather_schema_valid():
    """Garante que um DataFrame climático correto passa pelo validador."""
    df_valid = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-24 12:00:00")],
        "temperature_2m": [30.1],
        "wind_speed_100m": [15.2],
        "wind_direction_100m": [180.0]
    })
    validated_df = WeatherSchema.validate(df_valid)
    assert not validated_df.empty

def test_weather_schema_strict_rejection():
    """Garante que o schema rejeita colunas extras não declaradas (strict=True)."""
    df_extra_col = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-24 12:00:00")],
        "temperature_2m": [30.1],
        "wind_speed_100m": [15.2],
        "wind_direction_100m": [180.0],
        "coluna_intrusa": [999]  # Coluna extra que deve quebrar o strict
    })
    
    # Usando pa.errors.SchemaErrors (plural)
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        WeatherSchema.validate(df_extra_col)
        
    # Verifica se a coluna barrada é mencionada na mensagem de erro
    assert "coluna_intrusa" in str(exc_info.value)

def test_energy_schema_negative_generation():
    """Garante que geração de energia negativa é bloqueada no EnergySchema."""
    df_invalid_energy = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-24 12:00:00")],
        "wind_generation_mw": [-10.5]  # Abaixo de ge=0.0
    })
    with pytest.raises(pa.errors.SchemaError):
        EnergySchema.validate(df_invalid_energy)