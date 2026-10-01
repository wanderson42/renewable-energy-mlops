from unittest.mock import patch

import pandas as pd
import pytest

import energy_mlops.pipelines.backfill_flow as backfill_module


def make_checkpoint_dataframe(
    available_from: str = "2024-03-21",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "available_from": pd.to_datetime(
                [available_from]
            ),
        }
    )


def make_merged_dataframe(
    dates: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                dates,
                utc=True,
            ),
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
            "wind_generation_mw": [
                5000.0
                for _ in dates
            ],
        }
    )


def test_filter_capacity_covered_period_removes_pre_checkpoint_rows():
    df = make_merged_dataframe(
        [
            "2024-03-20 23:00:00",
            "2024-03-21 00:00:00",
            "2024-03-21 01:00:00",
        ]
    )

    checkpoints = (
        make_checkpoint_dataframe()
    )

    with patch.object(
        backfill_module,
        "load_bahia_wind_capacity_checkpoints",
        return_value=checkpoints,
    ):
        result = (
            backfill_module
            .filter_capacity_covered_period(
                df
            )
        )

    assert len(result) == 2

    assert result[
        "date"
    ].tolist() == [
        pd.Timestamp(
            "2024-03-21 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2024-03-21 01:00:00",
            tz="UTC",
        ),
    ]


def test_filter_capacity_covered_period_preserves_exact_available_from():
    df = make_merged_dataframe(
        [
            "2024-03-21 00:00:00",
        ]
    )

    checkpoints = (
        make_checkpoint_dataframe()
    )

    with patch.object(
        backfill_module,
        "load_bahia_wind_capacity_checkpoints",
        return_value=checkpoints,
    ):
        result = (
            backfill_module
            .filter_capacity_covered_period(
                df
            )
        )

    assert len(result) == 1

    assert result.loc[
        0,
        "date",
    ] == pd.Timestamp(
        "2024-03-21 00:00:00",
        tz="UTC",
    )


def test_filter_capacity_covered_period_does_not_backfill():
    """
    Uma observação anterior ao primeiro available_from
    deve desaparecer, e não receber capacidade futura.
    """

    df = make_merged_dataframe(
        [
            "2024-03-01 00:00:00",
            "2024-03-20 23:00:00",
            "2024-03-21 00:00:00",
        ]
    )

    checkpoints = (
        make_checkpoint_dataframe()
    )

    with patch.object(
        backfill_module,
        "load_bahia_wind_capacity_checkpoints",
        return_value=checkpoints,
    ):
        result = (
            backfill_module
            .filter_capacity_covered_period(
                df
            )
        )

    assert len(result) == 1

    assert (
        result["date"].min()
        == pd.Timestamp(
            "2024-03-21 00:00:00",
            tz="UTC",
        )
    )


def test_filter_capacity_covered_period_rejects_no_covered_rows():
    df = make_merged_dataframe(
        [
            "2024-03-01 00:00:00",
            "2024-03-20 23:00:00",
        ]
    )

    checkpoints = (
        make_checkpoint_dataframe()
    )

    with (
        patch.object(
            backfill_module,
            "load_bahia_wind_capacity_checkpoints",
            return_value=checkpoints,
        ),
        pytest.raises(
            ValueError,
            match=(
                "Nenhuma observação possui "
                "cobertura causal"
            ),
        ),
    ):
        (
            backfill_module
            .filter_capacity_covered_period(
                df
            )
        )


def test_historical_backfill_propagates_month_failure():
    """
    Uma falha mensal deve interromper o backfill.

    O pipeline não pode publicar silenciosamente um
    Gold contendo apenas os meses que tiveram sucesso.
    """

    with (
        patch.object(
            backfill_module,
            "extract_weather_task",
            side_effect=RuntimeError(
                "Falha no mês 2024-03"
            ),
        ),
        pytest.raises(
            RuntimeError,
            match="Falha no mês 2024-03",
        ),
    ):
        (
            backfill_module
            .historical_backfill_flow
            .fn(
                start_year=2024,
                end_year=2024,
                start_month=3,
                end_month=3,
            )
        )