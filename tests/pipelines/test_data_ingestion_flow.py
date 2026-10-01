import unittest.mock

import pandas as pd
import pytest

import energy_mlops.pipelines.data_ingestion_flow as ingestion_module


def make_energy_dataframe(
    dates: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "wind_generation_mw": [
                1000.0
                for _ in dates
            ],
        }
    )


def make_weather_dataframe(
    dates: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "temperature_2m": [
                25.0
                for _ in dates
            ],
            "wind_speed_100m": [
                10.0
                for _ in dates
            ],
            "wind_direction_100m": [
                180.0
                for _ in dates
            ],
        }
    )


def test_validate_energy_month_boundary_accepts_complete_month():
    """
    Setembro/2026 termina em 30/09 23:00
    America/Sao_Paulo = 01/10 02:00 UTC.
    """

    df_energy = make_energy_dataframe(
        [
            "2026-09-01 03:00:00",
            "2026-10-01 02:00:00",
        ]
    )

    ingestion_module.validate_energy_month_boundary(
        df_energy,
        year=2026,
        month=9,
    )


def test_validate_energy_month_boundary_rejects_truncated_month():
    """
    Reproduz o caso observado na fonte ONS:
    setembro/2026 disponível somente até
    29/09 23:00 local = 30/09 02:00 UTC.
    """

    df_energy = make_energy_dataframe(
        [
            "2026-09-01 03:00:00",
            "2026-09-30 02:00:00",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Mês ONS ainda incompleto",
    ):
        ingestion_module.validate_energy_month_boundary(
            df_energy,
            year=2026,
            month=9,
        )


def test_validate_energy_month_boundary_allows_internal_gaps():
    """
    A validação verifica fechamento mensal.

    Lacunas internas podem existir porque horas
    incompletas são removidas pelo extractor ONS.
    """

    df_energy = make_energy_dataframe(
        [
            "2025-01-01 03:00:00",
            "2025-01-02 02:00:00",
            # 03/01 ausente por data quality.
            "2025-01-04 03:00:00",
            "2025-02-01 02:00:00",
        ]
    )

    ingestion_module.validate_energy_month_boundary(
        df_energy,
        year=2025,
        month=1,
    )


def test_validate_energy_month_boundary_rejects_empty_dataframe():
    df_energy = pd.DataFrame(
        columns=[
            "date",
            "wind_generation_mw",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Dataset ONS vazio",
    ):
        ingestion_module.validate_energy_month_boundary(
            df_energy,
            year=2026,
            month=9,
        )


def test_data_ingestion_flow_validates_energy_boundary():
    """
    O flow mensal deve validar que o ONS alcançou
    o fechamento do mês antes de publicar o Gold.
    """

    dates = [
        "2026-09-01 03:00:00",
        "2026-10-01 02:00:00",
    ]

    df_energy = make_energy_dataframe(
        dates
    )

    df_weather = make_weather_dataframe(
        dates
    )

    df_merged = pd.merge(
        df_weather,
        df_energy,
        on="date",
        how="inner",
    )

    df_transformed = (
        df_merged.copy()
    )

    with (
        unittest.mock.patch.object(
            ingestion_module,
            "extract_weather_task",
            return_value=df_weather,
        ),
        unittest.mock.patch.object(
            ingestion_module,
            "extract_energy_task",
            return_value=df_energy,
        ),
        unittest.mock.patch.object(
            ingestion_module,
            "validate_energy_month_boundary",
        ) as mock_validate,
        unittest.mock.patch.object(
            ingestion_module,
            "merge_datasets_task",
            return_value=df_merged,
        ),
        unittest.mock.patch.object(
            ingestion_module,
            "transform_features_task",
            return_value=df_transformed,
        ),
        unittest.mock.patch.object(
            ingestion_module,
            "save_dataset_to_lake_or_local",
            return_value=(
                "s3://energy-lake/"
                "gold/test.parquet"
            ),
        ),
    ):
        result = (
            ingestion_module
            .data_ingestion_flow
            .fn(
                year=2026,
                month=9,
            )
        )

    mock_validate.assert_called_once()

    args, kwargs = (
        mock_validate.call_args
    )

    if kwargs:
        assert kwargs["year"] == 2026
        assert kwargs["month"] == 9
        assert (
            kwargs["df_energy"]
            is df_energy
        )
    else:
        assert args[0] is df_energy
        assert args[1] == 2026
        assert args[2] == 9

    assert result == (
        "s3://energy-lake/"
        "gold/test.parquet"
    )


def test_extract_energy_task_propagates_extractor_failure():
    """
    O task não deve transformar uma falha da fonte
    em None.
    """

    with (
        unittest.mock.patch.object(
            ingestion_module,
            "fetch_ons_wind_generation",
            side_effect=RuntimeError(
                "ONS indisponível"
            ),
        ),
        pytest.raises(
            RuntimeError,
            match="ONS indisponível",
        ),
    ):
        ingestion_module.extract_energy_task.fn(
            year=2026,
            month=9,
        )


def test_extract_weather_task_propagates_extractor_failure():
    """
    O task não deve transformar uma falha da fonte
    em None.
    """

    with (
        unittest.mock.patch.object(
            ingestion_module,
            "fetch_open_meteo_wind_data",
            side_effect=RuntimeError(
                "Open-Meteo indisponível"
            ),
        ),
        pytest.raises(
            RuntimeError,
            match="Open-Meteo indisponível",
        ),
    ):
        ingestion_module.extract_weather_task.fn(
            start_date="2026-09-01",
            end_date="2026-09-30",
        )