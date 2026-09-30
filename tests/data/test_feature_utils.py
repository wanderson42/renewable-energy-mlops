import pandas as pd
import pytest

from energy_mlops.data.feature_utils import (
    get_model_feature_columns,
    select_model_features,
)


def test_get_model_feature_columns():
    columns = get_model_feature_columns()

    assert len(columns) == 9
    assert columns == [
        "temperature_2m",
        "wind_speed_100m",
        "wind_direction_100m",
        "wind_temp_ratio",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "wind_speed_roll_mean_3h",
    ]


def test_select_model_features():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"]),
        "temperature_2m": [25.0],
        "wind_speed_100m": [10.0],
        "wind_direction_100m": [180.0],
        "wind_temp_ratio": [0.4],
        "hour_sin": [0.0],
        "hour_cos": [1.0],
        "month_sin": [0.5],
        "month_cos": [0.866],
        "wind_speed_roll_mean_3h": [9.5],
        "target_fc": [0.5],
        "capacidade_mw": [10000.0],
        "wind_generation_mw": [5000.0],
    })

    X = select_model_features(df)

    assert X.columns.tolist() == get_model_feature_columns()
    assert X.shape == (1, 9)


def test_select_model_features_missing_column():
    df = pd.DataFrame({
        "temperature_2m": [25.0],
        "wind_speed_100m": [10.0],
    })

    with pytest.raises(ValueError, match="Features obrigatórias ausentes"):
        select_model_features(df)
