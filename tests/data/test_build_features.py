# Testes de unidade engenharia de features: src/energy_mlops/data/build_features.py
import numpy as np
import pandas as pd
import pytest

from energy_mlops.data.build_features import generate_wind_and_time_features


def make_capacity_checkpoints() -> pd.DataFrame:
    """Cria checkpoints mínimos para testes isolados de capacidade."""

    return pd.DataFrame(
        {
            "edition": [1, 2],
            "publication_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-03-01",
                ]
            ),
            "reference_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-03-01",
                ]
            ),
            "available_from": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-03-01",
                ]
            ),
            "capacity_mw": [
                10000.0,
                11000.0,
            ],
            "state": [
                "BA",
                "BA",
            ],
            "source": [
                "ABEEOLICA_INFOVENTO",
                "ABEEOLICA_INFOVENTO",
            ],
            "reported_reference_period": [
                "2025-01",
                "2025-03",
            ],
            "date_basis": [
                "explicit_source_reference",
                "explicit_source_reference",
            ],
            "source_url": [
                "https://example.com/1.pdf",
                "https://example.com/2.pdf",
            ],
            "notes": [
                "Checkpoint de teste.",
                "Checkpoint de teste.",
            ],
        }
    )


def test_capacity_is_step_function():
    """
    Garante que não exista interpolação entre checkpoints.

    O valor de fevereiro deve continuar sendo o último
    checkpoint conhecido em janeiro.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-02-15 00:00:00",
                    "2025-03-01 00:00:00",
                ]
            ),
            "temperature_2m": [
                25.0,
                25.0,
                25.0,
            ],
            "wind_speed_100m": [
                8.0,
                8.0,
                8.0,
            ],
            "wind_direction_100m": [
                180.0,
                180.0,
                180.0,
            ],
        }
    )

    result = generate_wind_and_time_features(
        df,
        capacity_checkpoints=make_capacity_checkpoints(),
    )

    assert result["capacidade_mw"].tolist() == [
        10000.0,
        10000.0,
        11000.0,
    ]


def test_capacity_checkpoint_is_not_used_before_available_from():
    """
    Garante que um checkpoint futuro não seja usado
    para preencher observações anteriores.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-12-31 23:00:00",
                ]
            ),
            "temperature_2m": [25.0],
            "wind_speed_100m": [8.0],
            "wind_direction_100m": [180.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Não existe checkpoint causal "
            "de capacidade"
        ),
    ):
        generate_wind_and_time_features(
            df,
            capacity_checkpoints=(
                make_capacity_checkpoints()
            ),
        )


def test_capacity_changes_only_when_checkpoint_becomes_available():
    """
    Garante que a nova capacidade passe a valer exatamente
    em available_from, e não antes.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-02-28 23:00:00",
                    "2025-03-01 00:00:00",
                ]
            ),
            "temperature_2m": [
                25.0,
                25.0,
            ],
            "wind_speed_100m": [
                8.0,
                8.0,
            ],
            "wind_direction_100m": [
                180.0,
                180.0,
            ],
        }
    )

    result = generate_wind_and_time_features(
        df,
        capacity_checkpoints=make_capacity_checkpoints(),
    )

    assert result["capacidade_mw"].tolist() == [
        10000.0,
        11000.0,
    ]


def test_generate_wind_and_time_features_physical_and_cyclical():
    """Valida os cálculos de derivadas físicas e variáveis cíclicas de tempo."""
    df_raw = pd.DataFrame({
        "date": ["2026-09-24 00:00:00", "2026-09-24 06:00:00"],
        "wind_speed_100m": [10.0, 20.0],
        "temperature_2m": [24.9, 19.9]
    })

    df_out = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)

    # 1. Validação da razão vento/temperatura: 10.0 / (24.9 + 0.1) = 10 / 25 = 0.4
    assert np.isclose(df_out.loc[0, "wind_temp_ratio"], 0.4)

    # 2. Validação das variáveis cíclicas de hora:
    # Hora 0 -> sin(0) = 0.0, cos(0) = 1.0
    assert np.isclose(df_out.loc[0, "hour_sin"], 0.0, atol=1e-5)
    assert np.isclose(df_out.loc[0, "hour_cos"], 1.0, atol=1e-5)

    # Hora 6 -> sin(2*pi*6/24) = sin(pi/2) = 1.0, cos(pi/2) = 0.0
    assert np.isclose(df_out.loc[1, "hour_sin"], 1.0, atol=1e-5)
    assert np.isclose(df_out.loc[1, "hour_cos"], 0.0, atol=1e-5)



def test_wind_speed_roll_mean_does_not_use_future_values():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-01 00:00:00",
                    "2026-08-01 01:00:00",
                    "2026-08-01 02:00:00",
                    "2026-08-01 03:00:00",
                ]
            ),
            "temperature_2m": [
                25.0,
                25.0,
                25.0,
                25.0,
            ],
            "wind_speed_100m": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "wind_direction_100m": [
                180.0,
                180.0,
                180.0,
                180.0,
            ],
        }
    )

    original = generate_wind_and_time_features(df)

    df_future_changed = df.copy()

    # Altera t2, que é futuro em relação a t0 e t1.
    df_future_changed.loc[
        2,
        "wind_speed_100m",
    ] = 1000.0

    changed = generate_wind_and_time_features(
        df_future_changed
    )

    assert (
        original.loc[0, "wind_speed_roll_mean_3h"]
        == changed.loc[0, "wind_speed_roll_mean_3h"]
    )

    assert (
        original.loc[1, "wind_speed_roll_mean_3h"]
        == changed.loc[1, "wind_speed_roll_mean_3h"]
    )


def test_wind_speed_roll_mean_3h_is_causal():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-01 00:00:00",
                    "2026-08-01 01:00:00",
                    "2026-08-01 02:00:00",
                    "2026-08-01 03:00:00",
                ]
            ),
            "temperature_2m": [
                25.0,
                25.0,
                25.0,
                25.0,
            ],
            "wind_speed_100m": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "wind_direction_100m": [
                180.0,
                180.0,
                180.0,
                180.0,
            ],
        }
    )

    result = generate_wind_and_time_features(df)

    expected = pd.Series(
        [
            10.0,
            15.0,
            20.0,
            30.0,
        ],
        name="wind_speed_roll_mean_3h",
    )

    pd.testing.assert_series_equal(
        result["wind_speed_roll_mean_3h"],
        expected,
        check_index_type=False,
    )

def test_target_fc_clipping():
    """Garante que a taxa de geração (target_fc) seja limitada entre 0.0 e 1.0."""
    df_raw = pd.DataFrame({
        "date": ["2026-09-24 12:00:00"],
        "wind_speed_100m": [10.0],
        "temperature_2m": [25.0],
        "wind_generation_mw": [999999.0]  # Valor absurdo para forçar o clip em 1.0
    })

    df_out = generate_wind_and_time_features(df_raw, use_exogenous_lags=False)

    assert df_out.loc[0, "target_fc"] == 1.0


def test_wind_speed_roll_mean_3h_respects_time_gaps():
    """
    Garante que a rolling de 3h não atravesse
    grandes lacunas temporais.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02 22:00:00",
                    "2025-01-02 23:00:00",
                    "2025-01-04 00:00:00",
                ]
            ),
            "temperature_2m": [
                25.0,
                25.0,
                25.0,
            ],
            "wind_speed_100m": [
                10.0,
                20.0,
                40.0,
            ],
            "wind_direction_100m": [
                180.0,
                180.0,
                180.0,
            ],
        }
    )

    result = generate_wind_and_time_features(
        df,
        capacity_checkpoints=(
            make_capacity_checkpoints()
        ),
    )

    expected = pd.Series(
        [
            10.0,
            15.0,
            40.0,
        ],
        name="wind_speed_roll_mean_3h",
    )

    pd.testing.assert_series_equal(
        result[
            "wind_speed_roll_mean_3h"
        ],
        expected,
        check_index_type=False,
    )