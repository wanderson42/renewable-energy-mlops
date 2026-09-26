import logging

import openmeteo_requests
import pandas as pd
import pandera.pandas as pa
import requests_cache
from retry_requests import retry

from energy_mlops.data.schema import WeatherSchema

# Configuração de log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Coordenadas do polo eólico de Morro do Chapéu - BA
DEFAULT_LATITUDE = -11.5503
DEFAULT_LONGITUDE = -41.1565


def fetch_open_meteo_wind_data(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame | None:
    """
    Extrai dados históricos de vento usando o cliente oficial otimizado do Open-Meteo.
    Retorna diretamente um DataFrame do Pandas.
    """
    logger.info("Configurando cliente Open-Meteo com cache e retries...")
    
    # Configura cache (1 hora) e retries (5 tentativas)
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Usando o endpoint de ARCHIVE (ao invés de FORECAST) para buscar histórico longo
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["wind_speed_100m", "wind_direction_100m", "temperature_2m"],
        "timezone": "America/Maceio",  # Ajuste para o fuso horário do Nordeste
    }

    logger.info(f"Requisitando dados de {start_date} a {end_date} (Lat: {latitude}, Lon: {longitude})")

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        logger.info("Processando dados binários para DataFrame...")
        hourly = response.Hourly()
        
        # O cliente exige respeitar a mesma ordem da lista "hourly" dos params
        hourly_wind_speed_100m = hourly.Variables(0).ValuesAsNumpy()
        hourly_wind_direction_100m = hourly.Variables(1).ValuesAsNumpy()
        hourly_temperature_2m = hourly.Variables(2).ValuesAsNumpy()

        # Montagem do DataFrame
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "wind_speed_100m": hourly_wind_speed_100m,
            "wind_direction_100m": hourly_wind_direction_100m,
            "temperature_2m": hourly_temperature_2m
        }

        df = pd.DataFrame(data=hourly_data)
        logger.info(f"Extração concluída. Validando contrato de dados ({len(df)} linhas)...")
        try:
            df_validated = WeatherSchema.validate(df)
            logger.info("Validação do Pandera concluída com sucesso. Dados íntegros.")
        except pa.errors.SchemaError as exc:
            logger.error(f"Falha no contrato de dados (Pandera): {exc}")
            # Em produção, o pipeline do Prefect pararia aqui enviando um alerta
            return None
        return df_validated

    except Exception as e:  # noqa: BLE001
        logger.error(f"Falha ao extrair dados climáticos: {e}")
        return None

if __name__ == "__main__":
    # Teste para extrair apenas o mês de junho de 2026 para validação
    df_clima = fetch_open_meteo_wind_data(start_date="2024-08-01", end_date="2024-08-30")
    
    if df_clima is not None:
        print("\nAmostra dos dados extraídos:\n")
        print(df_clima.head())
        print(df_clima.info())
     