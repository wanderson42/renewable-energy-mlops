import logging

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from energy_mlops.data.schema import (
    WeatherSchema,
)

logger = logging.getLogger(__name__)


# Proxy meteorológica regional utilizada pelo projeto:
# Morro do Chapéu, Bahia.
DEFAULT_LATITUDE = -11.5503
DEFAULT_LONGITUDE = -41.1565

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2025-12-31"

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/"
    "v1/archive"
)


def fetch_open_meteo_wind_data(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Extrai variáveis meteorológicas horárias
    do Open-Meteo Archive API.

    Por padrão utiliza Morro do Chapéu como
    proxy meteorológica regional para o sistema
    eólico da Bahia.

    Essa é uma aproximação espacial do projeto:
    o ponto não representa individualmente as
    condições meteorológicas de todos os parques
    eólicos do estado.

    O resultado é validado por WeatherSchema.
    """

    start_timestamp = pd.Timestamp(
        start_date
    )

    end_timestamp = pd.Timestamp(
        end_date
    )

    if start_timestamp > end_timestamp:
        raise ValueError(
            "start_date não pode ser "
            "posterior a end_date."
        )

    logger.info(
        "Consultando Open-Meteo de %s a %s "
        "(lat=%s, lon=%s).",
        start_date,
        end_date,
        latitude,
        longitude,
    )

    cache_session = (
        requests_cache.CachedSession(
            ".cache",
            expire_after=3600,
        )
    )

    retry_session = retry(
        cache_session,
        retries=5,
        backoff_factor=0.2,
    )

    openmeteo = (
        openmeteo_requests.Client(
            session=retry_session
        )
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "wind_speed_100m",
            "wind_direction_100m",
            "temperature_2m",
        ],
        "timezone": (
            "America/Maceio"
        ),
    }

    try:
        responses = (
            openmeteo.weather_api(
                OPEN_METEO_ARCHIVE_URL,
                params=params,
            )
        )

        if not responses:
            raise RuntimeError(
                "Open-Meteo não retornou "
                "nenhuma resposta."
            )

        response = responses[0]

        hourly = response.Hourly()

        wind_speed_100m = (
            hourly
            .Variables(0)
            .ValuesAsNumpy()
        )

        wind_direction_100m = (
            hourly
            .Variables(1)
            .ValuesAsNumpy()
        )

        temperature_2m = (
            hourly
            .Variables(2)
            .ValuesAsNumpy()
        )

        date_index = pd.date_range(
            start=pd.to_datetime(
                hourly.Time(),
                unit="s",
                utc=True,
            ),
            end=pd.to_datetime(
                hourly.TimeEnd(),
                unit="s",
                utc=True,
            ),
            freq=pd.Timedelta(
                seconds=hourly.Interval()
            ),
            inclusive="left",
        )

        expected_length = len(
            date_index
        )

        variable_lengths = {
            "wind_speed_100m": len(
                wind_speed_100m
            ),
            "wind_direction_100m": len(
                wind_direction_100m
            ),
            "temperature_2m": len(
                temperature_2m
            ),
        }

        invalid_lengths = {
            name: length
            for name, length
            in variable_lengths.items()
            if length != expected_length
        }

        if invalid_lengths:
            raise ValueError(
                "Comprimento inconsistente "
                "na resposta Open-Meteo. "
                f"Esperado={expected_length}, "
                f"recebido={invalid_lengths}."
            )

        df = pd.DataFrame(
            {
                "date": date_index,
                "wind_speed_100m": (
                    wind_speed_100m
                ),
                "wind_direction_100m": (
                    wind_direction_100m
                ),
                "temperature_2m": (
                    temperature_2m
                ),
            }
        )

        validated = (
            WeatherSchema.validate(df)
        )

        logger.info(
            "Extração Open-Meteo concluída: "
            "%s horas validadas.",
            len(validated),
        )

        return validated

    except Exception:
        logger.exception(
            "Falha ao extrair dados "
            "meteorológicos do Open-Meteo."
        )
        raise


if __name__ == "__main__":
    df_weather = (
        fetch_open_meteo_wind_data(
            start_date="2024-08-01",
            end_date="2024-08-30",
        )
    )

    print(
        "\nAmostra dos dados "
        "meteorológicos:\n"
    )

    print(df_weather.head())

    print()
    print(df_weather.info())