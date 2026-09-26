import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class WeatherSchema(pa.DataFrameModel):
    """Contrato de dados para as variáveis climáticas (Features)."""
    
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    temperature_2m: Series[float] = pa.Field(ge=-50.0, le=60.0, nullable=False)

    # Focamos apenas no vento a 100m (altura do rotor da turbina eólica)
    wind_speed_100m: Series[float] = pa.Field(ge=0.0, le=200.0, nullable=False)    
    # Adicionando a direção do vento (em graus: de 0 a 360)
    wind_direction_100m: Series[float] = pa.Field(ge=0.0, le=360.0, nullable=False)

    class Config:
        coerce = True
        strict = True

class EnergySchema(pa.DataFrameModel):
    """Contrato de dados para a geração de energia do ONS."""
    
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    wind_generation_mw: Series[float] = pa.Field(ge=0.0, le=30000.0, nullable=False)

    class Config:
        coerce = True
        strict = True