import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class WeatherSchema(pa.DataFrameModel):
    """Contrato de dados para as Features climáticas."""

    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    temperature_2m: Series[float] = pa.Field(
        ge=-50.0,
        le=60.0,
        nullable=False,
    )
    wind_speed_100m: Series[float] = pa.Field(
        ge=0.0,
        le=200.0,
        nullable=False,
    )
    wind_direction_100m: Series[float] = pa.Field(
        ge=0.0,
        le=360.0,
        nullable=False,
    )

    class Config:
        coerce = True
        strict = True


class EnergySchema(pa.DataFrameModel):
    """Contrato de dados para a geração de energia do ONS."""

    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    wind_generation_mw: Series[float] = pa.Field(
        ge=0.0,
        le=30000.0,
        nullable=False,
    )

    class Config:
        coerce = True
        strict = True


class ModelFeatureSchema(pa.DataFrameModel):
    """Contrato das features finais utilizadas pelo modelo."""

    temperature_2m: Series[float] = pa.Field(nullable=False)
    wind_speed_100m: Series[float] = pa.Field(
        ge=0.0,
        le=200.0,
        nullable=False,
    )
    wind_direction_100m: Series[float] = pa.Field(
        ge=0.0,
        le=360.0,
        nullable=False,
    )

    wind_temp_ratio: Series[float] = pa.Field(nullable=False)

    hour_sin: Series[float] = pa.Field(nullable=False)
    hour_cos: Series[float] = pa.Field(nullable=False)
    month_sin: Series[float] = pa.Field(nullable=False)
    month_cos: Series[float] = pa.Field(nullable=False)

    wind_speed_roll_mean_3h: Series[float] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = True