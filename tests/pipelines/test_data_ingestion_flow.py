from unittest.mock import patch

import pandas as pd
import pytest

import energy_mlops.pipelines.data_ingestion_flow as ingestion_module

TEST_YEAR = 2026
TEST_MONTH = 9

COMPLETE_MONTH_DATES = [
    "2026-09-01 03:00:00",
    "2026-10-01 02:00:00",
]

EXPECTED_GOLD_PATH = (
    "s3://energy-lake/"
    "gold/test.parquet"
)


def make_energy_dataframe(
    dates: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "wind_generation_mw": (
                [1000.0] * len(dates)
            ),
        }
    )


def make_weather_dataframe(
    dates: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "temperature_2m": (
                [25.0] * len(dates)
            ),
            "wind_speed_100m": (
                [10.0] * len(dates)
            ),
            "wind_direction_100m": (
                [180.0] * len(dates)
            ),
        }
    )


def test_validate_energy_month_boundary_accepts_complete_month():
    """
    Setembro/2026 termina em 30/09 23:00
    America/Sao_Paulo = 01/10 02:00 UTC.
    """

    df_energy = make_energy_dataframe(
        COMPLETE_MONTH_DATES
    )

    ingestion_module.validate_energy_month_boundary(
        df_energy,
        year=TEST_YEAR,
        month=TEST_MONTH,
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
            year=TEST_YEAR,
            month=TEST_MONTH,
        )


def test_validate_energy_month_boundary_allows_internal_gaps():
    """
    A validação verifica o fechamento mensal.

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
            year=TEST_YEAR,
            month=TEST_MONTH,
        )


def test_data_ingestion_flow_validates_and_audits_before_persisting():
    """
    O flow mensal deve:

    1. validar o fechamento mensal do ONS;
    2. realizar merge e feature engineering;
    3. auditar o Gold transformado;
    4. persistir exatamente o DataFrame auditado.
    """

    df_energy = make_energy_dataframe(
        COMPLETE_MONTH_DATES
    )

    df_weather = make_weather_dataframe(
        COMPLETE_MONTH_DATES
    )

    df_merged = pd.merge(
        df_weather,
        df_energy,
        on="date",
        how="inner",
    )

    df_transformed = df_merged.copy()

    with (
        patch.object(
            ingestion_module,
            "extract_weather_task",
            return_value=df_weather,
        ),
        patch.object(
            ingestion_module,
            "extract_energy_task",
            return_value=df_energy,
        ),
        patch.object(
            ingestion_module,
            "validate_energy_month_boundary",
        ) as mock_validate,
        patch.object(
            ingestion_module,
            "merge_datasets_task",
            return_value=df_merged,
        ),
        patch.object(
            ingestion_module,
            "transform_features_task",
            return_value=df_transformed,
        ),
        patch.object(
            ingestion_module,
            "audit_gold_snapshot_task",
        ) as mock_audit,
        patch.object(
            ingestion_module,
            "save_dataset_to_lake_or_local",
            return_value=EXPECTED_GOLD_PATH,
        ) as mock_save,
    ):
        result = (
            ingestion_module
            .data_ingestion_flow
            .fn(
                year=TEST_YEAR,
                month=TEST_MONTH,
            )
        )

    mock_validate.assert_called_once_with(
        df_energy=df_energy,
        year=TEST_YEAR,
        month=TEST_MONTH,
    )

    mock_audit.assert_called_once()

    assert (
        mock_audit.call_args.args[0]
        is df_transformed
    )

    mock_save.assert_called_once()

    save_args, save_kwargs = (
        mock_save.call_args
    )

    persisted_df = (
        save_kwargs["df"]
        if "df" in save_kwargs
        else save_args[0]
    )

    assert persisted_df is df_transformed

    assert result == EXPECTED_GOLD_PATH


def test_extract_energy_task_propagates_extractor_failure():
    """
    A task não deve transformar uma falha da fonte
    ONS em None ou continuar silenciosamente.
    """

    with (
        patch.object(
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
            year=TEST_YEAR,
            month=TEST_MONTH,
        )


def test_extract_weather_task_propagates_extractor_failure():
    """
    A task não deve transformar uma falha
    da Open-Meteo em None.
    """

    with (
        patch.object(
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