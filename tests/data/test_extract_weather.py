from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import energy_mlops.data.extract_weather as weather_module


def _make_open_meteo_client(
    wind_speed: list[float] | None = None,
    wind_direction: list[float] | None = None,
    temperature: list[float] | None = None,
) -> MagicMock:
    """
    Constrói uma resposta Open-Meteo mínima
    compatível com a API utilizada pelo extractor.
    """

    if wind_speed is None:
        wind_speed = [
            10.0,
            12.0,
            14.0,
        ]

    if wind_direction is None:
        wind_direction = [
            120.0,
            130.0,
            140.0,
        ]

    if temperature is None:
        temperature = [
            20.0,
            21.0,
            22.0,
        ]

    start = int(
        pd.Timestamp(
            "2025-01-01 03:00:00",
            tz="UTC",
        ).timestamp()
    )

    interval = 3600

    hourly = MagicMock()

    hourly.Time.return_value = start

    hourly.TimeEnd.return_value = (
        start
        + 3 * interval
    )

    hourly.Interval.return_value = (
        interval
    )

    variables = []

    for values in [
        wind_speed,
        wind_direction,
        temperature,
    ]:
        variable = MagicMock()

        variable.ValuesAsNumpy.return_value = (
            np.asarray(
                values,
                dtype=float,
            )
        )

        variables.append(
            variable
        )

    hourly.Variables.side_effect = (
        lambda index: variables[index]
    )

    response = MagicMock()

    response.Hourly.return_value = (
        hourly
    )

    client = MagicMock()

    client.weather_api.return_value = [
        response
    ]

    return client


def test_fetch_open_meteo_returns_valid_dataframe():
    fake_client = (
        _make_open_meteo_client()
    )

    with (
        patch.object(
            weather_module.requests_cache,
            "CachedSession",
            return_value=MagicMock(),
        ),
        patch.object(
            weather_module,
            "retry",
            return_value=MagicMock(),
        ),
        patch.object(
            weather_module.openmeteo_requests,
            "Client",
            return_value=fake_client,
        ),
    ):
        result = (
            weather_module
            .fetch_open_meteo_wind_data(
                start_date="2025-01-01",
                end_date="2025-01-01",
            )
        )

    assert len(result) == 3

    assert result.columns.tolist() == [
        "date",
        "wind_speed_100m",
        "wind_direction_100m",
        "temperature_2m",
    ]

    assert result[
        "wind_speed_100m"
    ].tolist() == [
        10.0,
        12.0,
        14.0,
    ]

    assert result[
        "wind_direction_100m"
    ].tolist() == [
        120.0,
        130.0,
        140.0,
    ]

    assert result[
        "temperature_2m"
    ].tolist() == [
        20.0,
        21.0,
        22.0,
    ]


def test_fetch_open_meteo_rejects_invalid_date_range():
    with pytest.raises(
        ValueError,
        match=(
            "start_date não pode ser "
            "posterior a end_date"
        ),
    ):
        (
            weather_module
            .fetch_open_meteo_wind_data(
                start_date="2025-02-01",
                end_date="2025-01-01",
            )
        )


def test_fetch_open_meteo_rejects_inconsistent_lengths():
    fake_client = (
        _make_open_meteo_client(
            wind_speed=[
                10.0,
                12.0,
            ],
        )
    )

    with (
        patch.object(
            weather_module.requests_cache,
            "CachedSession",
            return_value=MagicMock(),
        ),
        patch.object(
            weather_module,
            "retry",
            return_value=MagicMock(),
        ),
        patch.object(
            weather_module.openmeteo_requests,
            "Client",
            return_value=fake_client,
        ),
        pytest.raises(
            ValueError,
            match=(
                "Comprimento inconsistente"
            ),
        ),
    ):
        (
            weather_module
            .fetch_open_meteo_wind_data(
                start_date="2025-01-01",
                end_date="2025-01-01",
            )
        )


def test_fetch_open_meteo_propagates_api_failure():
    fake_client = MagicMock()

    fake_client.weather_api.side_effect = (
        RuntimeError(
            "Open-Meteo indisponível"
        )
    )

    with (
        patch.object(
            weather_module.requests_cache,
            "CachedSession",
            return_value=MagicMock(),
        ),
        patch.object(
            weather_module,
            "retry",
            return_value=MagicMock(),
        ),
        patch.object(
            weather_module.openmeteo_requests,
            "Client",
            return_value=fake_client,
        ),
        pytest.raises(
            RuntimeError,
            match="Open-Meteo indisponível",
        ),
    ):
        (
            weather_module
            .fetch_open_meteo_wind_data(
                start_date="2025-01-01",
                end_date="2025-01-01",
            )
        )